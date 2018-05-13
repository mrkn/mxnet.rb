module MXNet
  module Gluon
    # Applies an `Optimizer` on a set of Parameters.
    # Trainer should be used together with `Autograd`.
    class Trainer
      def initialize(params, optimizer, optimizer_params: nil, kvstore: :device, compression_params: nil)
        case params
        when Hash, ParameterHash
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
          @params << param
        end
        @compression_params = compression_params
        optimizer_params ||= {}
        @scale = optimizer_params[:rescale_grad] || 1.0
        @contexts = check_contexts
        init_optimizer(optimizer, optimizer_params)
        @kv_initialized = false
        @kvstore = kvstore
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
        param_hash = @params.map.with_index {|x, i| [i, x] }.to_h
        if optimizer.is_a? Optimizer
          raise ArgumentError,
            "optimizer_params must be nil if optimizer is an instance of " +
            "Optimizer instead of str" if optimizer_params
          @optimizer = optimizer
          @optimizer.param_hash = param_hash
        else
          @optimizer = Optimizer[optimizer].new(param_hash: param_hash, **optimizer_params)
        end

        @updaters = @contexts.map { Optimizer.get_updater(@optimizer) }
      end

      private def init_kvstore
        arg_arrays = @params.map {|param| [param.name, param.data(@contexts[0])]}.to_h
        kvstore, update_on_kvstore = create_kvstore(@kvstore, @contexts.length, arg_arrays)
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
    end

    def step(batch_size, ignore_stale_grad: false)
      init_kvstore unless @kv_initialized

      @optimizer.rescale_grad = @scale / batch_size

      @params.each_with_index do |param, i|
        continue if param.grad_req == 'null'
        unless ignore_stale_grad
          param.list_data.each do |data|
            unless data.fresh_grad
              raise "Gradient of Parameter `#{param.name}` on context #{data.context} " +
                    "has not been updated by backward since last `step`. This could " +
                    "mean a bug in your model that maked it only use a subset of the " +
                    "Parameters (Blocks) for this iteration. If you are intentionally " +
                    "only using a subset, call step with ignore_stale_grad=True to " +
                    "suppress this warning and skip updating of Parameters with " +
                    "stale gradient"
            end
          end
        end

        if @kvstore
          @kvstore.push(i, param.list_grad, priority: -i)
          if @update_on_kvstore
            @kvstore.pull(i, param.list_data, priority: -i)
            continue
          else
            @kvstore.pull(i, param.list_grad, priority: -i)
          end
        end

        @updaters.zip(param.list_data, param.list_grad).each do |upd, arr, grad|
          if !ignore_stale_grad || arr.fresh_grad
            upd.(i, grad, arr)
            arr.fresh_grad = false
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

# TODO: KVStore
# TODO: 