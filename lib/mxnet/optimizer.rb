module MXNet
  module Optimizer
    module Registry
      @opt_registry = {}

      # Registers a new optimizer.
      #
      # Once an optimizer is registered, we can create an instance of this
      # optimizer with `Optimizer[name].new` later.
      def register(klass)
        unless klass < MXNet::Optimizer::Base
          raise ArgumentError, "optimizer must be a subclass of MXNet::Optimizer::Base"
        end
        name = self.name[/::([^:]+)\z/, 1].downcase.to_sym
        if @opt_registry.has_key? name
          warn "WARNING: New optimizer #{self} is overriding existing " +
              "optimizer #{@opt_registry[name]}"
        end
        @opt_registry[name] = self
      end

      def [](name)
        @opt_registry[name.downcase.to_sym] or
          raise KeyError, "Cannot find optimizer #{name}"
      end
    end

    extend Registry

    # The base class inherited by all optimizers.
    class Base
      def initialize(rescale_grad: 1.0,
                     param_idx2name: nil,
                     wd: 0.0,
                     clip_gradient: nil,
                     learning_rate: 0.01,
                     lr_scheduler: nil,
                     sym: nil,
                     begin_num_update: 0,
                     multi_precision: false,
                     param_dict: nil)
        @rescale_grad = rescale_grad
        @lr = learning_rate
        @lr_scheduler = lr_scheduler
        @lr_scheduler.base_lr = learning_rate if @lr_scheduler

        @wd = wd
        @lr_mult = {}
        @wd_mult = {}
        @begin_num_update = begin_num_update
        @num_update = begin_num_update
        @index_update_count = {}
        @clip_gradient = clip_gradient
        @multi_precision = multi_precision

        param_idx2name ||= {}
        unless param_idx2name.is_a? Hash
          raise ArgumentError, "param_idx2name should be a Hash of param indexes to names."
        end
        @idx2name = param_idx2name.dup
        @sym_info = sym ? [sym.attr_dict, sym.list_arguments] : []
        @param_dict = param_dict || {}

        self.lr_mult = {}
        self.wd_mult = {}
      end

      attr_reader :lr_scheduler

      def learning_rate
        if @lr_scheduler
          @lr_scheduler.(@num_update)
        else
          @lr
        end
      end

      # Sets a new learning rate of the optimizer.
      def learning_rate=(lr)
        if @lr_scheduler
          raise "LRScheduler of the optimizer has already been defined. " +
                "Note that learning_rate= can mutate the value of " +
                "the learning rate of the optimizer only when " +
                "the LRScheduler of the optimizer is undefined."
        else
          @lr = lr
        end
      end

      # Creates auxiliary state for a given weight.
      def create_state(index, weight)
        nil
      end

      # Creates auxiliary state for a given weight, including FP32 high
      # precision copy if original weight is FP16.
      #
      # This method is provided to perform automatic mixed precision training
      # for optimizers that do not support it themselves.
      def create_state_multi_precision(index, weight)
        weight_master_copy = nil
        if @multi_precision && weight.dtype == :float16
          weight_master_copy = weight.as_type(:float32)
          return [weight_master_copy, create_state(index, weight_master_copy)]
        end
        if weight.dtype == :float16 && !multi_precision
          warn "Accumulating with float16 in optimizer can lead to " +
               "poor accuracy or slow convergence. " +
               "Consider using multi_precision: true option of the optimizer"
        end
        create_state(index, weight)
      end

      # Updates the given parameter using the coressponding gradient and state.
      def update(index, weight, grad, state)
        raise NotImplementedError
      end

      # Updates the given parameter using the coressponding gradient and state.
      # Mixed precision version.
      def update_multi_precision(index, weight, grad, state)
        if @multi_precision && weight.dtype == :float16
          weight_master_copy = state[0]
          original_state = state[1]
          grad32 = grad.as_type(:float32)
          update(index, weight_master_copy, grad32, original_state)
          MXNet::NDArray.cast(weight, dtype: weight.dtype, out: weight)
        else
          update(index, weight, grad, state)
        end
      end

      # Sets an individual learning rate multiplier for each parameter.
      def lr_mult=(args_lr_mult)
        @lr_mult = {}
        if not @sym_info.empty?
          attrs, arg_names = @sym_info
          arg_names.each do |name|
            if attrs.has_key?(name) && attrs[name].has_key?(:__lr_mult__)
              @lr_mult[name] = Float(attrs[name][:__lr_mult__])
            end
          end
        end
        @lr_mult.update(args_lr_mult)
      end

      # Sets an individual weight decay multiplier for each parameter.
      def wd_mult=(args_wd_mult)
        @wd_mult = {}
        @idx2name.each_value do |n|
          @wd_mult[n] = 0.0 if n.end_with?('_weight') || n.end_with?('_gamma')
        end
        if not @sym_info.empty?
          attrs, arg_names = @sym_info
          arg_names.each do |name|
            if attrs.has_key?(name) && attrs[name].has_key?(:__wd_mult__)
              @wd_mult[name] = Float(attrs[name][:__wd_mult__])
            end
          end
        end
        @wd_mult.update(args_wd_mult)
      end

      # Updates num_update.
      private def update_count(index)
        @index_update_count[index] ||= @begin_num_update
        @index_update_count[index] += 1
        @num_update = [@index_update_count[index], @num_update].max
      end

      # Gets the learning rate given the index of the weight.
      private def get_lr(index)
        if @lr_scheduler
          lr = @lr_scheduler.(@num_update)
        else
          lr = @lr
        end

        if @param_hash.has_key? index
          lr *= @param_hash[index].lr_mult
        elsif @lr_mult.has_key? index
          lr *= @lr_mult[index]
        elsif @idx2name.has_key? index
          lr *= @lr_mult[@idx2name[index]] || 1.0
        end
        return lr
      end

      # Gets weight decay for index.
      # Returns 0 for non-weights if the name of weights are provided for `__init__`.
      def get_wd(index)
        wd = @wd
        if @param_hash.has_key? index
          wd *= @param_hash[index].wd_mult
        elsif @wd_mult.has_key? index
          wd *= @wd_mult[index]
        elsif @idx2name.has_key? index
          wd *= @wd_mult[@idx2name[index]] || 1.0
        end
        return wd
      end
    end

    # The SGD optimizer with momentum and weight decay.
    class SGD < Base
      def initialize(momentum: 0.0, lazy_update: true, **kwargs)
        super(**kwargs)
        @momentum = momentum
        @lazy_update = lazy_update
      end

      def create_state_multi_precision(index, weight)
        weight_master_copy = nil
        if @multi_precision && weight.dtype == :float16
          weight_master_copy = weight.as_type(:float32)
          return [create_state(index, weight_master_copy), weight_master_copy]
        end
        if weight.dtype == :float16 && !multi_precision
          warn "Accumulating with float16 in optimizer can lead to " +
               "poor accuracy or slow convergence. " +
               "Consider using multi_precision: true option of the SGD optimizer"
        end
        return create_state(index, weight)
      end

      def create_state(index, weight)
        momentum = nil
        # TODO: stype = @lazy_update ? weight.stype : :default
        if @momentum != 0.0
          momentum = MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype) # stype: stype
        end
        return momentum
      end

      private def update_impl(index, weight, grad, state, multi_precision: false)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count(index)
        lr = get_lr(index)
        wd = get_wd(index)

        kwargs = { rescale_grad: @rescale_grad }
        kwargs[:momentum] = @momentum if @momentum > 0
        kwargs[:clip_gradient] = @clip_gradient if @clip_gradient

        if !multi_precision
          if state
            MXNet::NDArray.sgd_mom_update(weight, grad, state, out: weight, lr: lr, wd: wd, **kwargs)
          else
            MXNet::NDArray.sgd_update(weight, grad, out: weight, lr: lr, wd: wd, **kwargs)
          end
        else
          if state[0]
            MXNet::NDArray.mp_sgd_mom_update(weight, grad, state[0], state[1], out: weight, lr: lr, wd: wd, **kwargs)
          else
            MXNet::NDArray.mp_sgd_update(weight, grad, state[1], out: weight, lr: lr, wd: wd, **kwargs)
          end
        end
      end

      def update(index, weight, grad, state)
        update_impl(index, weight, grad, state, multi_precision: false)
        nil
      end

      def update_multi_precision(index, weight, grad, state)
        use_multi_precision = @multi_precision && weight.dtype == :float16
        update_impl(index, weight, grad, state, multi_precision: use_multi_precision)
        nil
      end
    end

    # TODO: Signum

    # TODO: FTML

    # TODO: DCASGD

    # TODO: NAG

    # TODO: SGLD

    # TODO: CCSGD

    # TODO: Adam

    # TODO: AdamGrad

    # TODO: RMSProp

    # TODO: AdaDelta

    # TODO: Ftrl

    # TODO: Adamax

    # TODO: Nadam

    class Test < Base
      def initialize(**kwargs)
        super
      end

      def create_state(index, weight)
        MXNet::NDArray.zeros(weight,shape, weight.context)
      end

      def update(index, weight, grad, state)
        weight[0..-1] += grad * @rescale_grad
        state[0..-1] = weight
        nil
      end
    end

    # Updater for kvstore.
    class Updater
      def initialize(optimizer)
        @optimizer = optimizer
        @states = {}
        @states_synced = {}
      end

      # Updates weight given gradient and index.
      def call(index, grad, weight)
        if @state.has_key? index
          @states[index] = @optimizer.create_state_multi_precision(index, weight)
          @states_synced[index] = true
        elsif !@states_synced[index]
          @states[index] = sync_state_context(@states[index], weight.context)
          @states_synced[index] = true
        end
        @optimizer.update_multi_precision(index, weight, grad, @states[index])
      end

      def sync_state_context(state, context)
        case state
        when MXNet::NDArray
          state.as_in_context(context)
        when Array
          state.map {|i| sync_state_context(i, context) }
        else
          state
        end
      end

      def states(dump_optimizer: false)
        if dump_optimizer
          Marshal.dump([@states, @optimizer])
        else
          Marshal.dump(@states)
        end
      end

      def states=(states)
        states = Marshal.load(states)
        if states.is_a?(Array) && states.length == 2
          @states, @optimizer = *states
        else
          @states = states
        end
        @states_synced = @states.keys.map {|k| [k, false] }.to_h
        nil
      end
    end
  end
end
