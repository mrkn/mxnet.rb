module MXNet
  module Gluon
    # Applies an `Optimizer` on a set of Parameters.
    # Trainer should be used together with `Autograd`.
    class Trainer
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +params+::           (ParameterDict)
      #                      The set of parameters to optimize.
      # +optimizer+::        (Optimizer)
      #                      The optimizer to use.
      # +optimizer_params+:: (Hash)
      #                      Key-word arguments to be passed to
      #                      optimizer constructor. For example,
      #                      <tt>{'learning_rate': 0.1}</tt>. See each
      #                      optimizer's constructor for a list of
      #                      additional supported arguments.
      # +kvstore+::          ...
      # +compression_params+:: ...
      def initialize(params, optimizer, optimizer_params: nil, kvstore: :device, compression_params: nil)
        case params
        when Hash, ParameterDict
          params = params.values
        end
        begin
          params = params.to_ary
        rescue TypeError
          raise ArgumentError, "First argument must be an Array or a Hash of Parameters, got #{params.class}"
        end
        @params = []
        params.each do |param|
          unless param.is_a? Parameter
            raise ArgumentError, "First argument must be an Array or a Hash of Parameters, get an Array of #{param.class}"
          end
          param._trainer = self
          @params << param
        end
        @compression_params = compression_params
        optimizer_params ||= {}
        @scale = optimizer_params[:rescale_grad] || 1.0
        @contexts = check_contexts
        init_optimizer(optimizer, optimizer_params)
        @kv_initialized = false
        @kvstore = kvstore
        @update_on_kvstore = nil
        @distributed = nil # TODO:
      end

      private def check_contexts
        @params[0].list_ctx.tap do |contexts0|
          if (param = @params.find {|param| contexts0 != param.list_ctx })
            raise ArgumentError,
              "All Parameters must be initialized on the same set of contexts, " +
              "but Parameter #{param.name} is initialized on #{param.list_ctx} " +
              "while previous Parameters are initialized on #{contexts0}."
          end
        end
      end

      private def init_optimizer(optimizer, optimizer_params)
        param_dict = @params.map.with_index {|x, i| [i, x] }.to_h
        if optimizer.is_a? Optimizer::Base
          raise ArgumentError,
            "optimizer_params must be empty if optimizer is an instance of " +
            "Optimizer instead of str" unless optimizer_params.empty?
          @optimizer = optimizer
          @optimizer.param_dict = param_dict
        else
          @optimizer = Optimizer.create(optimizer, param_dict: param_dict, **optimizer_params)
        end

        @updaters = @contexts.map { Optimizer::Updater.new(@optimizer) }
      end

      private def init_kvstore
        arg_arrays = @params.map {|param| [param.name, param.data(ctx: @contexts[0])]}.to_h
        # TODO:
        # kvstore, update_on_kvstore = create_kvstore(@kvstore, @contexts.length, arg_arrays)
        kvstore, update_on_kvstore = nil, nil
        if kvstore
          kvstore.set_gradient_compression(@compression_params) if @compression_params
          # TODO: what is kvstore.type?
          update_on_kvstore = false if kvstore.type.include? :dist
          @params.each_with_index do |param, i|
            param_arrays = param.list_data
            kvstore.init(i, param_arrays[0])
            kvstore.pull(i, param_arrays, priority: -i)
          end
          kvstore.set_optimizer(@optimizer) if update_on_kvstore
          @kvstore = kvstore
          @update_on_kvstore = update_on_kvstore
        else
          @kvstore = nil
          @update_on_kvstore = nil
        end

        @kv_initialized = true
      end

      def learning_rate
        unless @optimizer.is_a? Optimizer
          raise "Optimizer has to be defined before its learning rate can be accessed"
        end
        @optimizer.learning_rate
      end

      # Sets a new learning rate of the optimizer
      def learning_rate=(lr)
        unless @optimizer.is_a? Optimizer
          raise "Optimizer has to be defined before its learning rate is mutated"
        end
        @optimizer.learning_rate = lr
      end

      # Makes one step of parameter update.
      #
      # ====Parameters
      #
      # +batch_size+:: (integer)
      #                Batch size of data processed. Gradient will be
      #                normalized by `1/batch_size`. Set this to 1 if
      #                you normalized loss manually with `loss =
      #                mean(loss)`.
      def step(batch_size, ignore_stale_grad: false)
        init_kvstore unless @kv_initialized

        @optimizer.rescale_grad = @scale / batch_size

        _all_reduce_grads
        _update(ignore_stale_grad)
      end

      private def _all_reduce_grads
        if @kvstore
          @params.each_with_index do |param, i|
            next if @param.grad_req == 'null'
            @kvstore.push(i, param.list_grad, priority: -i)
            unless @update_on_kvstore
              @kvstore.pull(i, param.list_grad, priority: -i, ignore_sparse: @distributed)
            end
          end
        end
      end

      # Makes one step of parameter update.
      #
      # ====Parameter
      #
      # +batch_size+:: (integer)
      #                Batch size of data processed. Gradient will be
      #                normalized by `1/batch_size`. Set this to 1 if
      #                you normalized loss manually with `loss =
      #                mean(loss)`.
      def update(batch_size, ignore_stale_grad: false)
        init_kvstore unless @kv_initialized

        # TODO:
        # if self._params_to_init:
        #     self._init_params()

        if @kvstore && @update_on_kvstore
          raise 'update() when parameters are updated on kvstore ' +
                'is not supported. Try setting `update_on_kvstore` ' +
                'to False when creating trainer.'
        end

        @optimizer.rescale_grad = @scale / batch_size
        _update(ignore_stale_grad)
      end

      private def _update(ignore_stale_grad)
        @params.each_with_index do |param, i|
          continue if param.grad_req == 'null'

          unless ignore_stale_grad
            param.list_data.each do |data|
              unless data._fresh_grad
                raise "Gradient of Parameter `#{param.name}` on context #{data.context} " +
                      "has not been updated by backward since last `step`. This could " +
                      "mean a bug in your model that maked it only use a subset of the " +
                      "Parameters (Blocks) for this iteration. If you are intentionally " +
                      "only using a subset, call step with ignore_stale_grad=True to " +
                      "suppress this warning and skip updating of Parameters with " +
                      "stale gradient"
              end
            end
          end  # unless ignore_stale_grad

          if @kvstore && @update_on_kvstore
            # TODO:
            # if param._stype == 'default'
            #   # 'row_sparse' parameters are not pulled immediately - they're pulled
            #   # in `Block.forward`
            @kvstore.pull(i, param.list_data, priority: -i)
          end

          @updaters.zip(param.list_data, param.list_grad).each do |upd, arr, grad|
            if !ignore_stale_grad || arr._fresh_grad
              upd.(i, grad, arr)
              arr._fresh_grad = false
            end
          end
        end
      end

      # Saves trainer states (e.g. optimizer, momentum) to a file.
      def save_states(fname)
        raise "assertion failed: @optimizer.nil?" if @optimizer.nil?
        if @update_on_kvstore
          @kvstore.save_optimizer_states(fname, dump_optimizer: true)
        else
          IO.binwrite(fname, @updaters[0].get_states(dump_optimizer: true))
        end
      end

      # Loads trainer states (e.g. optimizer, momentum) from a file.
      def load_states(fname)
        init_kvstore unless @kv_initialized

        if @update_on_kvstore
          @kvstore.load_optimizer_states(fname)
          @optimizer = @kvstore.updater.optimizer
        else
          states = IO.binread(fname)
          @updaters.each do |updater|
            updater.set_states(states)
            updater.optimizer = @updaters[0].optimizer
          end
          @optimizer = @updaters[0].optimizer
        end
      end
    end
  end
end

# TODO: KVStore
# TODO: 
