module MXNet
  module Optimizer
    def self.registry_manager
      @registry_manager ||= MXNet::Registry::Manager.new(Base, :optimizer)
    end
    private_class_method :registry_manager

    def self.create(*args, **kwargs)
      registry_manager.create(*args, **kwargs)
    end

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

    # The base class inherited by all optimizers.
    #
    # Custom optimizers can be created by subclassing MXNet::Optimizer::Base
    # and implementing the required function #update. By default, the created
    # optimizer will be registered under its simplified class name
    # (`class.name.split('::').last.downcase.to_sym`) but it may be registered
    # under another name by calling MXNet::Optimizer.register.
    #
    #     class MyOptimizer < Optimizer
    #       def update(index, weight, gradient, state)
    #         ...
    #       end
    #     end
    #     register MyOptimizer, :myopt
    #
    class Base
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +rescale_grad+::  (float, optional)
      #                   Before updating, multiply the gradient with
      #                   "rescale_grad". Often choose to be
      #                   <tt>1.0/batch_size</tt>.
      # +learning_rate+:: (float, optional)
      #                   The initial learning rate.
      # +wd+::            (float, optional)
      #                   The weight decay (or L2 regularization)
      #                   coefficient. Modifies objective by adding a
      #                   penalty for having large weights.
      #
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

        @lr_mult = {}
        @wd_mult = {}
      end

      attr_accessor :rescale_grad, :param_dict

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
      #
      # ====Parameters
      #
      # +index+::    (integer)
      #              The unique index of the parameter into the
      #              individual learning rates and weight
      #              decays. Learning rates and weight decay may be set
      #              via #set_lr_mult and #set_wd_mult, respectively.
      # +weight+::   (NDArray)
      #              The parameter to be updated.
      # +gradient+:: (NDArray)
      #              The gradient of the objective with respect to this
      #              parameter.
      # +state+::    (any)
      #              The state returned by #create_state.
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
      #
      # ====Parameters
      #
      # +index+:: (integer)
      #           The index corresponding to the weight.
      #
      # ====Returns
      #
      # Learning rate for this index.
      private def get_lr(index)
        if @lr_scheduler
          lr = @lr_scheduler.(@num_update)
        else
          lr = @lr
        end

        if @param_dict.has_key? index
          lr *= @param_dict[index].lr_mult
        elsif @lr_mult.has_key? index
          lr *= @lr_mult[index]
        elsif @idx2name.has_key? index
          lr *= @lr_mult[@idx2name[index]] || 1.0
        end
        return lr
      end

      # Gets weight decay for index.
      # Returns 0 for non-weights if the name of weights are provided for `__init__`.
      #
      # ====Parameters
      #
      # +index+:: (integer)
      #           The index corresponding to the weight.
      #
      # ====Returns
      #
      # Weight decay for this index.
      private def get_wd(index)
        wd = @wd
        if @param_dict.has_key? index
          wd *= @param_dict[index].wd_mult
        elsif @wd_mult.has_key? index
          wd *= @wd_mult[index]
        elsif @idx2name.has_key? index
          wd *= @wd_mult[@idx2name[index]] || 1.0
        end
        return wd
      end

      # Gets the learning rates given the indices of the weights.
  
      # Parameters
      # ----------
      # +indices+:: list of integer
      #     Indices corresponding to weights.

      # Returns
      # -------
      # +lrs+:: list of float
      #     Learning rates for those indices.
      
      private def get_lrs(indices)
          
          if @lr_scheduler.nil?
            lr = @lr
          else
            lr = @lr_scheduler[@num_update]
          end

          lrs = indices.map{lr}

          indices.each_with_index do |index, i|
              if @param_dict.has_key? index
                  lrs[i] *= @param_dict[index].lr_mult
              elsif  @lr_mult.has_key? index
                  lrs[i] *= @lr_mult[index]
              elsif @idx2name.has_key? index
                  lrs[i] *= @lr_mult[@idx2name[index]] || 1.0
              end
          end

          lrs
      end
      
      # Gets weight decays for indices.
      # Returns 0 for non-weights if the name of weights are provided for `__init__`.

      # Parameters
      # ----------
      # +indices+:: list of int
      #     Indices of weights.

      # Returns
      # -------
      # +wds+:: list of float
      #     Weight decays for those indices.
      private def get_wds indices
        wds = indices.map {@wd}
        indices.each_with_index do |index, i|
          if @param_dict.has_key? index
              wds[i] *= @param_dict[index].wd_mult
          elsif @wd_mult.has_key? index 
              wds[i] *= @wd_mult[index]
          elsif @idx2name.has_key? index
              wds[i] *= @wd_mult[@idx2name[index]] || 1.0
          end
        end
        wds
      end

    end

    # The SGD optimizer with momentum and weight decay.
    class SGD < Base
      # Creates a new instance.
      #
      # This optimizer accepts the following parameters in addition to
      # those accepted by Optimizer.
      #
      # ====Parameters
      #
      # +momentum+:: (float, optional)
      #              The momentum value.
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

    registry_manager.register SGD

    # The Signum optimizer that takes the sign of gradient or momentum.

    # The optimizer updates the weight by::

    #     rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
    #     state = momentum * state + (1-momentum)*rescaled_grad
    #     weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    # References
    # ----------
    # Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli & Anima Anandkumar. (2018).
    # signSGD: Compressed Optimisation for Non-Convex Problems. In ICML'18.

    # See: https://arxiv.org/abs/1802.04434

    # For details of the update algorithm see
    # :class:`~mxnet.ndarray.signsgd_update` and :class:`~mxnet.ndarray.signum_update`.

    # This optimizer accepts the following parameters in addition to those accepted
    # by :class:`.Optimizer`.

    # Parameters
    # ----------
    # +momentum+:: float, optional
    #    The momentum value.
    # +wd_lh+:: float, optional
    #    The amount of decoupled weight decay regularization, see details in the original paper at:\
    #    https://arxiv.org/abs/1711.05101

    class Signum < Base
      def initialize learning_rate: 0.01, momentum: 0.9, wd_lh: 0.0, **kwargs
        super **kwargs

        @momentum = momentum
        @wd_lh = wd_lh
      end

      def create_state index, weight
        momentum = nil
        if @momentum != 0.0
          momentum = MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)
        end

        momentum
      end



      def update index, weight, grad, state
          update_impl index, weight, grad, state
      end

      private def update_impl index, weight, grad, state
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count index
        lr = get_lr index
        wd = get_wd index

        kwargs = {rescale_grad: @rescale_grad}
        kwargs[:momentum] = @momentum if @momentum > 0

        kwargs[:clip_gradient] = @clip_gradient if @clip_gradient

        kwargs[:wd_lh] = @wd_lh if @wd_lh


        if state.nil?
          MXNet::NDArray.signsgd_update weight, grad, out: weight,
                         lr: lr, wd: wd, **kwargs
        else
          MXNet::NDArray.signum_update weight, grad, state, out: weight,
                        lr: lr, wd: wd, **kwargs
        end
      end
    end

    registry_manager.register Signum

    class FTML < Base
      # The FTML optimizer.

      # This class implements the optimizer described in
      # *FTML - Follow the Moving Leader in Deep Learning*,
      # available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

      # Denote time step by t. The optimizer updates the weight by::

      #     rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
      #     v = beta2 * v + (1 - beta2) * square(rescaled_grad)
      #     d_t = (1 - power(beta1, t)) / lr * square_root(v / (1 - power(beta2, t))) + epsilon)
      #     z = beta1 * z + (1 - beta1) * rescaled_grad - (d_t - beta1 * d_(t-1)) * weight
      #     weight = - z / d_t

      # For details of the update algorithm, see :class:`~mxnet.ndarray.ftml_update`.

      # This optimizer accepts the following parameters in addition to those accepted
      # by :class:`.Optimizer`.

      # Parameters
      # ----------
      # +beta1+:: float, optional
      #     0 < beta1 < 1. Generally close to 0.5.
      # +beta2+:: float, optional
      #     0 < beta2 < 1. Generally close to 1.
      # +epsilon+:: float, optional
      #     Small value to avoid division by 0.

      def initialize beta1: 0.6, beta2: 0.999, epsilon: 1e-8, **kwargs
        super **kwargs
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon
      end

      def create_state index, weight
          [zeros(weight.shape, weight.context, dtype: weight.dtype), # d_0
           zeros(weight.shape, weight.context, dtype: weight.dtype), # v_0
           zeros(weight.shape, weight.context, dtype: weight.dtype)] # z_0
      end

      def update index, weight, grad, state
        raise unless weight.is_a? MXNet::NDArray
        raise unless grad.is_a? MXNet::NDArray
        update_count(index)
        lr = get_lr(index)
        wd = get_wd(index)
        t = @index_update_count[index]

        kwargs = {beta1: @beta1, beta2: @beta2, epsilon: @epsilon,
                  rescale_grad: @rescale_grad, t: t}

        kwargs[:clip_grad] = @clip_gradient if @clip_gradient

        prev_d, prev_v, prev_z = state
        MXNet::NDArray.ftml_update(weight, grad, prev_d, prev_v, prev_z, out=weight,
                    lr=lr, wd=wd, **kwargs)
      end
    end

    registry_manager.register FTML

    # the LARS optimizer from 'Large Batch Training of Convolution Networks' \
    # (https://arxiv.org/abs/1708.03888)

    # Behave mostly like SGD with momentum and weight decay but is scaling \
    # adaptively the learning for each layer (except bias and batch norm parameters):
    # w_norm = L2norm(weights)
    # g_norm = L2norm(gradients)
    # if w_norm > 0 and g_norm > 0:
    #     lr_layer = lr * lr_mult * eta * w_norm / (g_norm + weight_decay * w_norm + eps)
    # else:
    #     lr_layer = lr * lr_mult

    class LARS < Base
      # Parameters
      # ----------
      # momentum : float, optional
      #     The momentum value.
      # +lazy_update+:: bool, optional
      #     Default is True. If True, lazy updates are applied \
      #     if the storage types of weight and grad are both ``row_sparse``.
      # +lars_eta+:: float, optional
      #     LARS coefficient used to scale the learning rate. Default set to 0.001.
      # +lars_epsilon+:: float, optional
      #     Optional epsilon in case of very small gradients. Default set to 0.
      # +momentum_correction+:: bool, optional
      #     If True scale momentum w.r.t global learning rate change (with an lr_scheduler) \
      #     as indicated in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour` \
      #     (https://arxiv.org/pdf/1706.02677.pdf)
      #     Default set to True.
      
      def initialize momentum: 0.0, lazy_update: true, eta: 0.001, eps: 0,
                   momentum_correction: true, **kwargs
          super(**kwargs)
          @momentum = momentum
          @momentum_correction = momentum_correction
          @lazy_update = lazy_update
          @aggregate_num = ENV['MXNET_OPTIMIZER_AGGREGATION_SIZE'] || 4
          @eta = eta
          @eps = eps
          @skip = 0
          @last_lr = nil
          @cur_lr = nil
      end
  
      # Gets the learning rates given the indices of the weights.
  
      # Parameters
      # ----------
      # +indices+:: list of int
      #     Indices corresponding to weights.

      # Returns
      # -------
      # +lrs+:: list of float
      #     Learning rates for those indices.
      def get_lrs indices
        @last_lr = @cur_lr unless @cur_lr.nil?

        if @lr_scheduler.nil?
          lr = @lr
        else
          lr = @lr_scheduler[@num_update]
        end

        @last_lr = lr if @cur_lr.nil?
            
        @cur_lr = lr

        lrs = indices.map{ lr }

        indices.each_with_index do |index, i|
          if @param_dict.include? index
              lrs[i] *= @param_dict[index].lr_mult
          elsif @lr_mult.include? index 
              lrs[i] *= @lr_mult[index]
          elsif @idx2name.include? index
              lrs[i] *= @lr_mult[@idx2name[index]] || 1.0
          end
        end
        lrs
      end

      def set_wd_mult args_wd_mult
          @wd_mult = {}
          idx2name.values.each do |n|
              is_weight = n.endswith('_weight')
              @wd_mult[n] = 0.0 unless is_weight
          end

          if @sym_info
              _attr, arg_names = @sym_info
              arg_names.each do |name|
                  if _attr.include? name  and _attr[name].include? '__wd_mult__'
                      @wd_mult[name] = float(_attr[name]['__wd_mult__'])
                  end
              end
          end
          @wd_mult.update(args_wd_mult)
      end

      def create_state_multi_precision index, weight
          weight_master_copy = nil
          if @multi_precision and weight.dtype == :float16
              weight_master_copy = weight.as_type(:float32)
              return [create_state(index, weight_master_copy), weight_master_copy]
          end

          if weight.dtype == :float16 and not @multi_precision
              warn "Accumulating with float16 in optimizer can lead to " +
                            "poor accuracy or slow convergence. " +
                            "Consider using multi_precision=True option of the " +
                            "SGD optimizer"
          end

          create_state(index, weight)
      end

      def create_state index, weight
          momentum = nil
          if @momentum != 0.0
              #stype = weight.stype if self.lazy_update else 'default' TODO: stype
              momentum = MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype) #, stype=stype)
          end
          momentum
      end

      # L2 Norm implementation
      def _l2norm v, rescale=false
          v = v.as_type(:float32)
          v *= @rescale_grad if rescale
          
          MXNet::NDArray.norm(v)[0].to_f
      end

      # Returns a scaling factor for the learning rate for this layer
      def _get_lars i, weight, g, lr, wd
          
          name = @idx2name.include? i ? @idx2name[i] : i.to_s

          return lr if ['gamma', 'beta','bias'].any? {|n| name.end_with?(n)}
  
          w_norm = _l2norm(weight)
          g_norm = _l2norm(g, rescale: true)
  
          if w_norm > 0.0 and g_norm > 0.0
              lars = @eta * w_norm/(g_norm + wd * w_norm + @eps)
          else
              lars = 1.0
          end

          lars * lr
      end

      def _update_impl indices, weights, grads, states, multi_precision: false
        aggregate = true

        unless indices.is_a? Array
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        end

        weights.zip(grads).each do |weight, grad|
          raise 'Weight must be NDArray' unless weight.is_a? NDArray
          raise 'Grad must be NDArray' unless grad.is_a? NDArray

          aggregate = aggregate #and TODO: Support stype
                        #weight.stype == 'default' and
                        #grad.stype == 'default') 
        end

        update_count(indices)
        lrs = get_lrs(indices)
        wds = get_wds(indices)

        kwargs = {rescale_grad: @rescale_grad}

        if @momentum > 0
          if (@momentum_correction and @last_lr != 0)
            kwargs[:momentum] = (@momentum * (@cur_lr / @last_lr))
          else
            kwargs[:momentum] = @momentum
          end  
        end

        
        kwargs[:clip_gradient] = @clip_gradient if @clip_gradient

        if aggregate
            nb_params = indices.length
            names =  indices.map{|i| @idx2name.has_key?(i) ? @idx2name[i] : i.to_s}
            lars_idx = nb_params.times.reject {|i| ['gamma', 'beta', 'bias'].any?{|n| names[i].end_with? n }}

            nb_lars = lars_idx.length
            no_lars_idx = nb_params.times.select {|i| ['gamma', 'beta', 'bias'].any?{|n| names[i].end_with? n }}

            cur_ctx = weights[0].context
            full_idx = lars_idx + no_lars_idx
            new_lrs = MXNet::NDArray.array(full_idx.map{|i| lrs[i]}, ctx: cur_ctx, dtype: :float32)
            new_wds = MXNet::NDArray.array(full_idx.map{|i| wds[i]}, ctx: cur_ctx, dtype: :float32)
            new_weights = full_idx.map{|i| weights[i] } 
            new_grads =  full_idx.map{|i| grads[i] } 
            new_states =  full_idx.map{|i| states[i] } 
            if nb_lars > 0
               lars_range = 0..(nb_lars-1)
                w_sum_sq = MXNet::NDArray.multi_sum_sq(*new_weights[lars_range], num_arrays: nb_lars)
                g_sum_sq = MXNet::NDArray.multi_sum_sq(*new_grads[lars_range], num_arrays: nb_lars)
                MXNet::NDArray.multi_lars(new_lrs[lars_range], w_sum_sq, g_sum_sq, new_wds[lars_range],
                            eta: @eta, eps: @eps, rescale_grad: @rescale_grad,
                            out: new_lrs[lars_range])
            end
            # Same than usual using preloaded sgd functions
            sidx = 0
            while sidx < indices.length
                eidx = sidx + new_weights[sidx..(sidx+@aggregate_num)].length-1

                if not @multi_precision
                    if @momentum > 0
                      MXNet::NDArray.preloaded_multi_sgd_mom_update(
                            *((new_weights[sidx..eidx].zip(new_grads[sidx..eidx],
                                                new_states[sidx..eidx])).flatten +
                              [new_lrs[sidx..eidx], new_wds[sidx..eidx]]),
                            out: new_weights[sidx..eidx],
                            num_weights: new_weights[sidx..eidx].length,
                            **kwargs)
                    else
                      MXNet::NDArray.preloaded_multi_sgd_update(
                            *( (new_weights[sidx..eidx].zip(
                                                new_grads[sidx..eidx])).flatten +
                              [new_lrs[sidx..eidx], new_wds[sidx..eidx]]),
                            out: new_weights[sidx..eidx],
                            num_weights: new_weights[sidx..eidx].length,
                            **kwargs)
                    end
                else
                    if @momentum > 0
                      splat_states = *new_states[sidx..eidx]
                      first_state = splat_states.shift
                      MXNet::NDArray.preloaded_multi_mp_sgd_mom_update(
                            *((new_weights[sidx..eidx]).zip(new_grads[sidx..eidx],
                                                *first_state.zip(splat_states)).flatten +
                              [new_lrs[sidx..eidx], new_wds[sidx..eidx]]),
                            out: new_weights[sidx..eidx],
                            num_weights: new_weights[sidx..eidx].length,
                            **kwargs)
                    else
                      splat_states = *new_states[sidx..eidx]
                      first_state = splat_states.shift
                      MXNet::NDArray.preloaded_multi_mp_sgd_update(
                            *((new_weights[sidx..eidx].zip( new_grads[sidx..eidx],
                                                first_state.zip(splat_states)[1])).flatten +
                              [new_lrs[sidx..eidx], new_wds[sidx..eidx]]),
                            out: new_weights[sidx..eidx],
                            num_weights: new_weights[sidx..eidx].length,
                            **kwargs)
                    end
                end
                sidx += @aggregate_num
            end
        else
          lrs = indices.zip(weights, grads, lrs, wds).map{|i, w, g, lr, wd| get_lars(i, w, g, lr, wd)}
          weights.zip(grads, states, lrs, wds).each do |weight, grad, state, lr, wd|
            if @multi_precision
              if state[0].nil?
                MXNet::NDArray.mp_sgd_update(weight, grad, state[1], out: weight,
                  lr: lr, wd: wd, **kwargs)
              else
                MXNet::NDArray.mp_sgd_mom_update(weight, grad, state[0], state[1], out: weight,
                  lr: lr, wd: wd, **kwargs)
                  
              end  
            else
              if state.nil?
                MXNet::NDArray.sgd_update(weight, grad, out: weight, lazy_update: @lazy_update,
                  lr: lr, wd: wd, **kwargs)
              else
                MXNet::NDArray.sgd_mom_update(weight, grad, state, out: weight,
                  lazy_update: @lazy_update, lr: lr, wd: wd, **kwargs)  
              end
            end
          end
        end
      end

      def update index, weight, grad, state
          _update_impl(index, weight, grad, state, multi_precision: false)
      end
  
      def update_multi_precision index, weight, grad, state
          if index.is_a? Array
            use_multi_precision = @multi_precision and weight[0].dtype == :float16 
          else 
            use_multi_precision = @multi_precision and weight.dtype == :float16
          end
          _update_impl(index, weight, grad, state,
                            multi_precision: use_multi_precision)
      end
    end


    registry_manager.register LARS

    # The Large Batch SGD optimizer with momentum and weight decay.
  
    # The optimizer updates the weight by::

    #     state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
    #     weight = weight - state

    # For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update`
    # and :class:`~mxnet.ndarray.sgd_mom_update`.
    # In addition to the SGD updates the LBSGD optimizer uses the LARS, Layer-wise
    # Adaptive Rate Scaling, algorithm to have a separate learning rate for each
    # layer of the network, which leads to better stability over large batch sizes.

    # This optimizer accepts the following parameters in addition to those accepted
    # by :class:`.Optimizer`.

    class LBSGD < Base
  
      # Parameters
      # ----------
      # momentum : float, optional
      #     The momentum value.
      # multi_precision: bool, optional
      #     Flag to control the internal precision of the optimizer.
      #     False: results in using the same precision as the weights (default),
      #     True: makes internal 32-bit copy of the weights and applies gradients
      #     in 32-bit precision even if actual weights used in the model have lower precision.
      #     Turning this on can improve convergence and accuracy when training with float16.
  
      # warmup_strategy: string ('linear', 'power2', 'sqrt'. , 'lars'   default : 'linear')
      # warmup_epochs: unsigned, default: 5
      # batch_scale:   unsigned, default: 1 (same as batch size * numworkers)
      # updates_per_epoch: updates_per_epoch (default: 32, Default might not reflect true number batches per epoch. Used for warmup.)
      # begin_epoch: unsigned, default 0, starting epoch.
      
      def initialize momentum: 0.0, multi_precision: false, warmup_strategy: :linear,
                   warmup_epochs: 5, batch_scale: 1, updates_per_epoch: 32, begin_epoch: 0, num_epochs: 60,
                   **kwargs
          super(**kwargs)
          #logging.info('Running Large-Batch SGD Algorithm')
          #logging.info('(Batch_scale=%f, warmup_epochs=%d, warmup_strategy=%s, updates_per_epoch=%d)',
          #             batch_scale, warmup_epochs, warmup_strategy, updates_per_epoch)
          @momentum = momentum
          @multi_precision = multi_precision
          # new user parameters for large batch
          @warmup_strategy = warmup_strategy
          @warmup_epochs = warmup_epochs
          @batch_scale = batch_scale
          @updates_per_epoch = updates_per_epoch
          @init_updates = begin_epoch * updates_per_epoch
          @num_epochs = num_epochs
          # addl internal usage parameters and storage
          @lbmult = 1
          @cumgrads = {}
          # for adaptive lr
          @adaptive = false
          @admult = 1  # adaptation constant
      end
  
      def create_state index, weight
          momentum = nil
          weight_master_copy = nil
          if @multi_precision and weight.dtype == :float16
            weight_master_copy = MXNet::NDArray.array(weight, ctx: weight.context, dtype: :float32)
            if @momentum != 0.0
                momentum = MXNet::NDArray.zeros(weight.shape, weight.context, dtype: :float32)
                                  #stype:weight.stype) #TODO stype
            end
            return [momentum, weight_master_copy]
          end

          if weight.dtype == :float16 and not @multi_precision
            warn("Accumulating with float16 in optimizer can lead to " +
                          "poor accuracy or slow convergence. " +
                          "Consider using multi_precision=True option of the " +
                          "SGD optimizer")
          end
          
          if @momentum != 0.0
              momentum = MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype) #, stype: weight.stype)
          end
          momentum
      end
  
      # Returns lr scaling factor for large batch according to warmup schedule
      # (to be implemented)
      def get_lbmult nup
        nwup = @warmup_epochs * @updates_per_epoch
        strategy = @warmup_strategy
        maxmult = @batch_scale.to_f
        if nup >= nwup
          mult = maxmult
        elsif nwup <= 1
          mult = 1.0
        else
          if strategy == :linear
              mult = 1.0 + (maxmult - 1) * nup / nwup
          elsif strategy == :power2
              mult = 1.0 + (maxmult-1) * (nup*nup)/(nwup*nwup)
          elsif strategy == :sqrt
              mult = 1.0 + (maxmult - 1) * Math.sqrt(float(nup) / nwup)
          else
              mult = 1.0
          end
        end
        mult
      end

      #Returns a scaling factor for the learning rate for this layer
      #default is 1
      def get_lars weight, g, wd
          
          weight2 = l2norm(weight)
          grad2 = l2norm(g)
          lars = Math.sqrt(weight2 / (grad2 + wd * weight2 + 1e-18))
          if lars < 0.01
            lars = 0.01
          elsif lars > 100
              lars = 100
          end
          lars
      end

      # inner product implementation
      def l2norm v
          
          norm = multiply(v, v).asnumpy().sum()
          return norm
      end

      # called every macro-batch to reset cumulated gradients to 0 for a given index
      def reset_cum_gradient index
          @cumgrads[index][:cum_grad] = 0
      end

      # get the cumulated gradient for index
      def get_cum_gradient index
          if @cumgrads.include? index
              return @cumgrads[index]
          else
              return {}
          end
      end

      # store cumulated gradient for index
      def put_cum_gradient index, cgrad
          @cumgrads[index] = cgrad
      end

      # Cumulate gradients for large-batch emulation. Cumulated by index (layer)
      def cumulate_gradient grad, index
          cgrad = get_cum_gradient(index)
          if cgrad
              num_cums = cgrad[:num_cums]
              if num_cums > 0
                  cum_grad = cgrad[:cum_grad] + grad
                  num_cums += 1
              else
                  cum_grad = grad
                  num_cums = @init_updates + 1
              end
          else
              cum_grad = grad
              num_cums = @init_updates + 1
          end

          cgrad = {cum_grad: cum_grad, num_cums: num_cums}
          put_cum_gradient(index, cgrad)
          cgrad
      end
      
      def update index, weight, grad, state
        raise 'weight must be an NDArray' unless weight.is_a? MXNet::NDArray
        raise 'grad must be an NDArray' unless grad.is_a? MXNet::NDArray
        

        lr = get_lr(index)
        wd = get_wd(index)
        update_count(index)

        # new stuff for large batch
        cgrad = cumulate_gradient(grad, index)
        if (cgrad[:num_cums] % batch_scale) == 0
            grad = cgrad[:cum_grad] / self.batch_scale
            if @warmup_strategy == :lars
                lbmult = get_lars(weight, grad, wd)
            else
                lbmult = get_lbmult(cgrad[:num_cums])
            end
            lr = lr * lbmult
            # do the regular sgd update flow
            kwargs = {rescale_grad: @rescale_grad}

            kwargs[:momentum] = @momentum if @momentum > 0
            kwargs[:clip_gradient] = @clip_gradient if @clip_gradient

            use_multi_precision = state.is_a? Array

            if use_multi_precision
              if state[0].nil?
                MXNet::NDArray.mp_sgd_update(weight, grad, state[1], out: weight, lr: lr, wd: wd, **kwargs)
              else
                MXNet::NDArray.mp_sgd_mom_update(weight, grad, state[0], state[1], out: weight, lr: lr, wd: wd, **kwargs)
              end 
            else
              if state.nil?
                MXNet::NDArray.sgd_update(weight, grad, out: weight, lr: lr, wd: wd, **kwargs)
              else
                MXNet::NDArray.sgd_mom_update(weight, grad, state, out: weight, lr: lr, wd: wd, **kwargs)
              end 
            end
            # reset update count and cumulated gradient per large batch
            reset_cum_gradient(index)
        else
            lr = 0.0
            kwargs = {}
            MXNet::NDArray.sgd_update(weight, grad, out: weight, lr: lr, wd: wd, **kwargs)
        end
      end
    end

    registry_manager.register LBSGD


    # LAMB Optimizer.
    # 
    class LAMB < Base
      def initialize learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-6,
                   lower_bound: nil, upper_bound: nil, bias_correction: true, **kwargs
          super(**kwargs)
          @beta1 = beta1
          @beta2 = beta2
          @epsilon = epsilon
          @lower_bound = lower_bound
          @upper_bound = upper_bound
          @bias_correction = bias_correction
          @aggregate_num = [1, [45, ENV['MXNET_OPTIMIZER_AGGREGATION_SIZE'] || 45].min].max
      end

      def create_state index, weight
          #stype = weight.stype TODO stype
          dtype = weight.dtype
          return [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: dtype), #, stype=stype), #TODO stype
            MXNet::NDArray.zeros(weight.shape, weight.context, dtype: dtype)] #, stype=stype)], #TODO stype
      end

      def _update_impl index, weight, grad, state, multi_precision: false
          kwargs = {beta1: @beta1, beta2: @beta2, epsilon: @epsilon,
            bias_correction: @bias_correction,
            rescale_grad: @rescale_grad}
  
          if @aggregate_num <= 1 or not index.is_a? Array
              if index.is_a? Array
                  raise 'Must have same number of indices as aggregate_num' unless index.length == @aggregate_num
                  index, weight, grad, state = index[0], weight[0], grad[0], state[0]
              end

              raise 'weight must be NDArray' unless weight.is_a? NDArray
              raise 'grad must be NDArray' unless grad.is_a? NDArray
              update_count(index)
              lr = get_lr(index)
              wd = get_wd(index)
              t = @index_update_count[index]
              
              weight_ptr = weight
              grad_ptr = grad
              if multi_precision
                  mean, var = state[1]
                  weight32 = state[0]
              else
                  mean, var = state
              end

              kwargs[:t] = t

              
              kwargs[:clip_gradient] = @clip_gradient if @clip_gradient
  
              if multi_precision
                  g = MXNet::NDArray.mp_lamb_update_phase1(weight_ptr, grad_ptr, mean, var, weight32, wd: wd, **kwargs)
                  kwargs = {}
                  kwargs[:lower_bound] = @lower_bound if @lower_bound
                  kwargs[:upper_bound] = @upper_bound if @upper_bound
                      
                  r_1 = MXNet::NDArray.norm(weight32)
                  r_2 = MXNet::NDArray.norm(g)
                  MXNet::NDArray.mp_lamb_update_phase2(weight_ptr, g, r_1, r_2, weight32, lr: lr, out: weight_ptr, **kwargs)
              else
                  g = MXNet::NDArray.lamb_update_phase1(weight_ptr, grad_ptr, mean, var, wd: wd, **kwargs)
                  kwargs = {}
                  
                  kwargs[:lower_bound] = @lower_bound if @lower_bound
              
                  kwargs[:upper_bound] = @upper_bound if @upper_bound
                  r_1 = MXNet::NDArray.norm(weight_ptr)
                  r_2 = MXNet::NDArray.norm(g)
                  MXNet::NDArray.lamb_update_phase2(weight_ptr, g, r_1, r_2, lr: lr, out: weight_ptr, **kwargs)
              end
          else
            kwargs[:clip_gradient] = @clip_gradient if @clip_gradient
            kwargs[:lower_bound] = @lower_bound if @lower_bound
                
            kwargs[:upper_bound] = @upper_bound if @upper_bound
                

            step_count, lrs, wds = [], [], []
            for i, w_i, g_i in index.zip(weight, grad)
              raise 'w_i must be NDArray' unless w_i.is_a? NDArray   
              raise 'g_i must be NDArray' unless g_i.is_a? NDArray   
              update_count(i)
              step_count.append(@index_update_count[i])
              lrs.append(get_lr(i))
              wds.append(get_wd(i))
            end

            updated_tensors = 0
            while updated_tensors < weight.length
                sidx = updated_tensors
                eidx = [updated_tensors + @aggregate_num, weight.length].min -1
                if multi_precision
                  mean_var = list(zip(*state[sidx..eidx]))[1]
                  temp = list(zip(*mean_var))
                  mean = temp[0]
                  var = temp[1]
                  splat_states = *state[sidx..eidx]
                  first_state = splat_states.shift
                  MXNet::NDArray.multi_mp_lamb_update(weight[sidx..eidx],
                                        grad[sidx..eidx],
                                        mean, var,
                                        Array(first_state.zip(splat_states))[0],
                                        out: weight[sidx..eidx],
                                        step_count: step_count[sidx..eidx],
                                        lrs: lrs[sidx..eidx],
                                        wds: wds[sidx..eidx],
                                        **kwargs)                    
                else
                  mean, var = list(zip(*state[sidx..eidx]))
                  MXNet::NDArray.multi_lamb_update(weight[sidx..eidx],
                                      grad[sidx..eidx],
                                      mean, var,
                                      out: weight[sidx..eidx],
                                      step_count: step_count[sidx..eidx],
                                      lrs: lrs[sidx..eidx],
                                      wds: wds[sidx..eidx],
                                      **kwargs)

                end
                updated_tensors += @aggregate_num
            end
          end
      end

      def update index, weight, grad, state
          _update_impl(index, weight, grad, state, multi_precision: false)
      end
      
      def update_multi_precision index, weight, grad, state
          if index.is_a? Array
            use_multi_precision = @multi_precision and weight[0].dtype == :float16
          else
            use_multi_precision = @multi_precision and weight.dtype == :float16
          end
          _update_impl(index, weight, grad, state, multi_precision: use_multi_precision)
      end

    end


    registry_manager.register LAMB

    # The DCASGD optimizer.

    # This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent
    # with Delay Compensation for Distributed Deep Learning*,
    # available at https://arxiv.org/abs/1609.08326.

    # This optimizer accepts the following parameters in addition to those accepted
    # by :class:`.Optimizer`.

    # Parameters
    # ----------
    # +momentum+:: float, optional
    #    The momentum value.

    # +lamda+:: float, optional
    #    Scale DC value.
    # 

    class DCASGD < Base
      def initialize momentum: 0.0, lamda: 0.04, **kwargs
          super **kwargs
          @momentum = momentum
          @weight_previous = {}
          @lamda = lamda
      end
      def create_state index, weight
          if @momentum == 0.0
            return [nil,
                    weight.copy_to(weight.context) ]  # previous weight
          else
            return [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype), # momentum
                    weight.copy_to(weight.context) ]  # previous weight
          end
      end

      def update index, weight, grad, state
          raise unless weight.is_a? MXNet::NDArray
          raise unless grad.is_a? MXNet::NDArray

          update_count(index)
          lr = get_lr(index)
          wd = get_wd(index)

          grad = grad * @rescale_grad
          grad = clip(grad, -@clip_gradient, @clip_gradient) unless @clip_gradient.nil?

          mom, previous_weight = state
          if mom
              mom *= @momentum
              mom += -lr * (grad + wd * weight + @lamda \
                               * grad * grad * (weight - previous_weight))
          else
              raise 'Momentum must be zero without mom' unless @momentum == 0.0
              mom = -lr * (grad + wd * weight + @lamda \
                           * grad * grad * (weight - previous_weight))
          end

          previous_weight = weight
          weight += mom
      end
    end

    registry_manager.register DCASGD



    # TODO: NAG

    class SGLD < Base
        # Stochastic Gradient Riemannian Langevin Dynamics.

        # This class implements the optimizer described in the paper *Stochastic Gradient
        # Riemannian Langevin Dynamics on the Probability Simplex*, available at
        # https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex.pdf.

        #
        def initialize **kwargs
            super(**kwargs)
        end

        def create_state index, weight
            return nil
        end

        def update index, weight, grad, state
            raise unless weight.is_a? MXNet::NDArray
            raise unless grad.is_a? MXNet::NDArray

            update_count(index)
            lr = get_lr(index)
            wd = get_wd(index)

            grad = grad * @rescale_grad
            grad = clip(grad, -@clip_gradient, @clip_gradient) unless @clip_gradient.nil?

            weight += - lr/2 * (grad + wd * weight)
            weight += MXNet::NDArray::Random.normal(0, Math.sqrt(lr), shape: weight.shape,
                                dtype: weight.dtype, ctx: weight.context)
        end
    end

    registry_manager.register SGLD

    # The Adam optimizer.

    # This class implements the optimizer described in *Adam: A Method for
    # Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.

    # If the storage types of grad is ``row_sparse``, and ``lazy_update`` is True, \
    # **lazy updates** at step t are applied by::

    #     for row in grad.indices:
    #         rescaled_grad[row] = clip(grad[row] * rescale_grad + wd * weight[row], clip_gradient)
    #         m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
    #         v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
    #         lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
    #         w[row] = w[row] - lr * m[row] / (sqrt(v[row]) + epsilon)

    # The lazy update only updates the mean and var for the weights whose row_sparse
    # gradient indices appear in the current batch, rather than updating it for all indices.
    # Compared with the original update, it can provide large improvements in model training
    # throughput for some applications. However, it provides slightly different semantics than
    # the original update, and may lead to different empirical results.

    # Otherwise, **standard updates** at step t are applied by::

    #     rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
    #     m = beta1 * m + (1 - beta1) * rescaled_grad
    #     v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
    #     lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
    #     w = w - lr * m / (sqrt(v) + epsilon)
    class Adam < Base

      # This optimizer accepts the following parameters in addition to those accepted
      # by :class:`.Optimizer`.

      # For details of the update algorithm, see :class:`~mxnet.ndarray.adam_update`.

      # Parameters
      # ----------
      # +beta1+:: float, optional
      #           Exponential decay rate for the first moment estimates.
      # +beta2+:: float, optional
      #           Exponential decay rate for the second moment estimates.
      # +epsilon+:: float, optional
      #           Small value to avoid division by 0.
      # +lazy_update+:: bool, optional
      #           Default is True. If True, lazy updates are applied \
      #           if the storage types of weight and grad are both ``row_sparse``.

      def initialize( beta1: 0.9, beta2: 0.99, epsilon: 1e-8, lazy_update: true,
                      weight: nil, batch_axis: 0, **kwargs)
        super(**kwargs)
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon
        @lazy_update = lazy_update

      end

      def create_state(index, weight)
        #TODO: stype =  @lazy_update ? weight.stype : :default
        [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype),  # mean, TODO: stype: stype
                      MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)]  # variance, TODO: stype: stype
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count(index)

        lr = get_lr(index)
        wd = get_wd(index)

        t = @index_update_count[index]
        coef1 = 1. - @beta1**t
        coef2 = 1. - @beta2**t
        lr = lr * Math.sqrt(coef2)/coef1

        kwargs = {beta1: @beta1, beta2: @beta2, epsilon: @epsilon,
                  rescale_grad: @rescale_grad}

        kwargs[:clip_gradient] = @clip_gradient if @clip_gradient

        mean, var = state
        MXNet::NDArray.adam_update(weight, grad, mean, var, out: weight,
                    lazy_update: @lazy_update, lr: lr, wd: wd, **kwargs)
      end
    end


    registry_manager.register Adam

    # AdaGrad optimizer.
    #
    #     This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    #     Methods for Online Learning and Stochastic Optimization*, and available at
    #     http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    #
    #     This optimizer updates each weight by::
    #
    #         grad = clip(grad * rescale_grad, clip_gradient)
    #         history += square(grad)
    #         div = grad / sqrt(history + float_stable_eps)
    #         weight += (div + weight * wd) * -lr
    #
    class AdaGrad < Base
      # This optimizer accepts the following parameters in addition to those accepted
      # by :class:`.Optimizer`.
      #
      # See Also
      # ----------
      # :meth:`mxnet.ndarray.sparse.adagrad_update`.
      #
      #     Parameters
      # ----------
      # eps: float, optional
      # Initial value of the history accumulator. Avoids division by 0.

      def initialize( epsilon: 1e-7, **kwargs)
        super(**kwargs)
        @float_stable_eps = epsilon
      end

      def create_state(index, weight)
        MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype) # TODO: stype: stype
      end


      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        is_sparse = grad.dtype == 'row_sparse' # TODO stype ?
        update_count(index)

        lr = get_lr(index)
        wd = get_wd(index)
        history = state

        if is_sparse

          kwargs = {epsilon: epsilon, rescale_grad: rescale_grad}
          if @clip_gradient
            kwargs[:clip_gradient] = @clip_gradient
          end
          # When grad is sparse, update weight with fused kernel
          MXNet::NDArray::Sparse.adagrad_update(weight, grad, history, out: weight, lr: lr, wd: wd, **kwargs)
        else
          grad *= rescale_grad
          if @clip_gradient
            grad = clip(grad, -@clip_gradient, @clip_gradient)
          end

          #update history
          history += MXNet::NDArray::square(grad)
          d = grad / (MXNet::NDArray::sqrt(history) + @float_stable_eps)

          # update weight
          weight += (d + weight * wd) -lr
        end
      end

    end

    registry_manager.register AdaGrad


    # The RMSProp optimizer.
    #
    # Two versions of RMSProp are implemented:
    #
    # If ``centered=False``, we follow
    # http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    # Tieleman & Hinton, 2012.
    # For details of the update algorithm see :class:`~mxnet.ndarray.rmsprop_update`.
    #
    # If ``centered=True``, we follow http://arxiv.org/pdf/1308.0850v5.pdf (38)-(45)
    # by Alex Graves, 2013.
    # For details of the update algorithm see :class:`~mxnet.ndarray.rmspropalex_update`.
    #
    class RMSProp < Base

      # This optimizer accepts the following parameters in addition to those accepted
      # by :class:`.Optimizer`.
      #
      # Parameters
      # ----------
      # gamma1: float, optional
      #     A decay factor of moving average over past squared gradient.
      # gamma2: float, optional
      #     A "momentum" factor. Only used if `centered`=``True``.
      # epsilon : float, optional
      #     Small value to avoid division by 0.
      # centered : bool, optional
      #     Flag to control which version of RMSProp to use.::
      #
      #         True: will use Graves's version of `RMSProp`,
      #         False: will use Tieleman & Hinton's version of `RMSProp`.
      #
      # clip_weights : float, optional
      #     Clips weights into range ``[-clip_weights, clip_weights]``.
      def initialize(learning_rate: 0.001, gamma1: 0.9, gamma2: 0.9,
                   epsilon: 1e-8, centered: false, clip_weights: nil, **kwargs)
        super(learning_rate: learning_rate, **kwargs)
        @gamma1 = gamma1
        @gamma2 = gamma2
        @centered = centered
        @epsilon = epsilon
        @clip_weights = clip_weights
      end

      def create_state(index, weight)
        if @centered
          [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype), # n TODO stype
           MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype), # g TODO stype
           MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)] # delta TODO stype
        else
          [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)] # n TODO stype
        end
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a?(NDArray)
        raise unless grad.is_a?(NDArray)
        update_count(index)
        lr = get_lr(index)
        wd = get_wd(index)

        kwargs = {gamma1: @gamma1, epsilon: @epsilon,
                  rescale_grad: @rescale_grad}
        if @centered
          kwargs[:gamma2] = @gamma2
        end
        if @clip_gradient
          kwargs[:clip_gradient] = @clip_gradient
        end
        if @clip_weights
          kwargs[:clip_weights] = @clip_weights
        end

        if @centered
          n, g, delta = state
          MXNet::NDArray.rmspropalex_update(weight, grad, n, g, delta, out: weight, lr: lr, wd: wd, **kwargs)
        else
          n, *rest = state
          MXNet::NDArray.rmsprop_update(weight, grad, n, out: weight, lr: lr, wd: wd, **kwargs)
        end
      end
    end

    registry_manager.register RMSProp


    # The AdaDelta optimizer.
    #
    #     This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    #     learning rate method*, available at https://arxiv.org/abs/1212.5701.
    #
    #         This optimizer updates each weight by::
    #
    #             grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
    #             acc_grad = rho * acc_grad + (1. - rho) * grad * grad
    #             delta = sqrt(acc_delta + epsilon) / sqrt(acc_grad + epsilon) * grad
    #             acc_delta = rho * acc_delta + (1. - rho) * delta * delta
    #             weight -= (delta + wd * weight)
    class AdaDelta < Base
      #     This optimizer accepts the following parameters in addition to those accepted
      #     by :class:`.Optimizer`.
      #
      #             Parameters
      #     ----------
      #     rho: float
      #     Decay rate for both squared gradients and delta.
      #         epsilon : float
      #     Small value to avoid division by 0.

      def initialize(rho: 0.90, epsilon: 1e-5, **kwargs)
        super(**kwargs)
        @rho=rho
        @epsilon=epsilon
      end

      def create_state(index, weight)
        [MXNet::NDArray.zeros(weight.shape, weight.context),  # accumulated g
         MXNet::NDArray.zeros(weight.shape, weight.context)] # accumulated delta
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray

        wd = get_wd(index)
        update_count(index)

        # # preprocess grad
        grad *= @rescale_grad
        if @clip_gradient
          grad = clip(grad, - @clip_gradient, @clip_gradient)
        end

        # accumulated g and delta initlization
        acc_g, acc_delta = state

        acc_g *= @rho
        acc_g += (1. - @rho) * grad * grad
        current_delta = MXNet::NDArray::sqrt(acc_delta + @epsilon) / MXNet::NDArray::sqrt(acc_g + @epsilon) * grad
        acc_delta *= @rho
        acc_delta += (1. - @rho) * current_delta * current_delta


        # update weight
        weight -= current_delta + wd * weight
      end

    end

    registry_manager.register AdaDelta

    # The Ftrl optimizer.
    #
    #     Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    #     http://dl.acm.org/citation.cfm?id=2488200.
    #
    #     eta :
    #         .. math::
    #            \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^2}}
    #
    #     The optimizer updates the weight by::
    #
    #         rescaled_grad = clip(grad * rescale_grad, clip_gradient)
    #         z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
    #         n += rescaled_grad**2
    #         w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)
    #
    #     If the storage types of weight, state and grad are all ``row_sparse``, \
    #     **sparse updates** are applied by::
    #
    #         for row in grad.indices:
    #             rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
    #             z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
    #             n[row] += rescaled_grad[row]**2
    #             w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)
    #
    #     The sparse update only updates the z and n for the weights whose row_sparse
    #     gradient indices appear in the current batch, rather than updating it for all
    #     indices. Compared with the original update, it can provide large
    #     improvements in model training throughput for some applications. However, it
    #     provides slightly different semantics than the original update, and
    #     may lead to different empirical results.
    #
    #     For details of the update algorithm, see :class:`~mxnet.ndarray.ftrl_update`.

    class Ftrl < Base

      #     This optimizer accepts the following parameters in addition to those accepted
      #     by :class:`.Optimizer`.
      #
      #     Parameters
      #     ----------
      #     lamda1 : float, optional
      #         L1 regularization coefficient.
      #     learning_rate : float, optional
      #         The initial learning rate.
      #     beta : float, optional
      #         Per-coordinate learning rate correlation parameter.
      #
      def initialize(lamda1=0.01, learning_rate=0.1, beta=1, **kwargs)
        super(**kwargs)
        @lamda1 = lamda1
        @beta = beta
        @lr = learning_rate
      end

      def create_state(index, weight)
        [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype),  # z #TODO stype: stype
         MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)]  # n #TODO stype: stype
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count(index)

        lr = get_lr(index)
        wd = get_wd(index)

        kwargs = {lamda1: @lamda1, beta: @beta, rescale_grad: @rescale_grad}
        if self.clip_gradient
          kwargs[:clip_gradient] = @clip_gradient
        end

        # accumulated g and delta initialization
        z, n = state
        ftrl_update(weight, grad, z, n, out=weight, lr=lr, wd=wd, **kwargs)
      end
    end

    registry_manager.register Ftrl

    # The AdaMax optimizer.
    #
    #     It is a variant of Adam based on the infinity norm
    #     available at http://arxiv.org/abs/1412.6980 Section 7.
    #
    #     The optimizer updates the weight by::
    #
    #         grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
    #         m = beta1 * m_t + (1 - beta1) * grad
    #         u = maximum(beta2 * u, abs(grad))
    #         weight -= lr / (1 - beta1**t) * m / u
    #
    class Adamax < Base

      def initialize(learning_rate: 0.002, beta1: 0.9, beta2: 0.999, **kwargs)
        super(learning_rate: learning_rate,**kwargs)
        @beta1=beta1
        @beta2=beta2
      end

      def create_state(index, weight)
        [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype),  # mean
         MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)]  # variance
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count(index)

        lr = get_lr(index)
        wd = get_wd(index)

        t = @index_update_count[index]
        lr /= (1. - @beta1**t)

        # preprocess grad
        grad = grad * @rescale_grad + wd * weight
        if @clip_gradient
          grad = clip(grad, -@clip_gradient, @clip_gradient)
        end
        # update m_t and u_t
        m_t, u_t = state
        m_t *= @beta1
        m_t += (1. - @beta1) * grad
        u_t = maximum(@beta2 * u_t, NDabs(grad))

        # update weight
        weight -= lr * m_t / u_t

      end
    end

    registry_manager.register Adamax

    # The Nesterov Adam optimizer.
    #
    #     Much like Adam is essentially RMSprop with momentum,
    #     Nadam is Adam RMSprop with Nesterov momentum available
    #     at http://cs229.stanford.edu/proj2015/054_report.pdf.
    #
    class Nadam < Base
      #     This optimizer accepts the following parameters in addition to those accepted
      #     by :class:`.Optimizer`.
      #
      #     Parameters
      #     ----------
      #     beta1 : float, optional
      #         Exponential decay rate for the first moment estimates.
      #     beta2 : float, optional
      #         Exponential decay rate for the second moment estimates.
      #     epsilon : float, optional
      #         Small value to avoid division by 0.
      #     schedule_decay : float, optional
      #         Exponential decay rate for the momentum schedule

      def initialize(learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, schedule_decay: 0.004, **kwargs)
        super(learning_rate: learning_rate, **kwargs)
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon
        @schedule_decay = schedule_decay
        @m_schedule = 1.0
      end

      def create_state(index, weight)
        [MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype),  # mean
         MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)]  # variance
      end

      def update(index, weight, grad, state)
        raise unless weight.is_a? NDArray
        raise unless grad.is_a? NDArray
        update_count(index)
        lr = get_lr(index)
        wd = get_wd(index)

        t = @index_update_count[index]

        # preprocess grad
        grad = grad * @rescale_grad + wd * weight
        if @clip_gradient
          grad = clip(grad, -@clip_gradient, @clip_gradient)

          # warming momentum schedule
          momentum_t = @beta1 * (1. - 0.5 * (pow(0.96, t * @schedule_decay)))
          momentum_t_1 = @beta1 * (1. - 0.5 * (pow(0.96, (t + 1) * @schedule_decay)))
          @m_schedule = @m_schedule * momentum_t
          m_schedule_next = @m_schedule * momentum_t_1

          # update m_t and v_t
          m_t, v_t = state
          m_t *= @beta1
          m_t += (1. - @beta1) * grad
          v_t *= @beta2
          v_t += (1. - @beta2) * grad * grad

          grad_prime = grad / (1. - @m_schedule)
          m_t_prime = m_t / (1. - m_schedule_next)
          v_t_prime = v_t / (1. - pow(@beta2, t))
          m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

          # update weight
          weight -= lr * m_t_bar / (sqrt(v_t_prime) + @epsilon)
        end
      end
    end

    registry_manager.register Nadam

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
        if !@states.has_key? index
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
