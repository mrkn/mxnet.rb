require 'set'

module MXNet
  module Gluon
    # Error for unfinished deferred initializations.
    class DeferredInitializationError < MXNet::Error
    end

    # A container holding parameters (weights) of blocks.
    #
    # Parameter holds a copy of the parameter on each Context after it
    # is initialized with #init. Also holds a gradient array on each
    # Context.
    #
    # ====Parameters
    #
    # +name+::  (string) Name of this parameter.
    # +shape+:: (integer or array of integers) Shape of this parameter.
    #           By default shape is not specified.
    #
    class Parameter
      def initialize(name, grad_req: :write, shape: nil, dtype: :float32,
                     lr_mult: 1.0, wd_mult: 1.0, initializer: nil,
                     allow_deferred_init: false, differentiable: true)
        @_var = nil
        @_data = nil
        @_grad = nil
        @_ctx_list = nil
        @_ctx_map = nil
        @_trainer = nil
        @_deferred_init = []
        @_differentiable = differentiable
        @_allow_deferred_init = allow_deferred_init
        @_grad_req = nil
        @_shape = shape && Array(shape)
        @name = name
        @dtype = dtype
        @lr_mult = lr_mult
        @wd_mult = wd_mult
        self.grad_req = grad_req
        @initializer = initializer
        @attributes = {}
      end

      attr_accessor :name, :dtype, :lr_mult, :wd_mult, :initializer

      attr_reader :_trainer

      # Set the trainer this parameter is associated with.
      def _trainer=(trainer)
        # TODO: trainer cannot be replaced for sparse params
        @_trainer = trainer
      end

      def has_attr?(attr_name)
        attr_name = attr_name.to_sym
        case attr_name
        when :name, :dtype, :lr_mult, :wd_mult, :initializer
          true
        else
          @attributes.has_key?(attr_name)
        end
      end

      def [](attr_name)
        attr_name = attr_name.to_sym
        case attr_name
        when :name, :dtype, :lr_mult, :wd_mult, :initializer
          send(attr_name)
        else
          @attributes[attr_name]
        end
      end

      def []=(attr_name, value)
        attr_name = attr_name.to_sym
        case attr_name
        when :name, :dtype, :lr_mult, :wd_mult, :initializer
          send(:"#{attr_name}=", value)
        else
          @attributes[attr_name] = value
        end
      end

      def inspect
        s = "Parameter #{name} (shape=#{shape}, dtype=#{dtype})"
      end

      VALID_GRAD_REQ = Set[:write, :add, :null]
      VALID_GRAD_REQ.merge(VALID_GRAD_REQ.map(&:to_s))
      VALID_GRAD_REQ.freeze

      def grad_req
        @_grad_req
      end

      def grad_req=(req)
        unless VALID_GRAD_REQ.include?(req)
          raise ArgumentError, "grad_req must be one of :write, :add, or :null, but got #{req}"
        end
        req = :null unless @_differentiable
        return req if @_grad_req == req
        @_grad_req = req
        if req == :null && @_grad
          @_grad = nil
          @_data = @_data.map(&:detach)
        elsif @_data
          _init_grad
        end
        req
      end

      def shape
        @_shape
      end

      def shape=(new_shape)
        return @_shape = new_shape unless @_shape

        compatible_p = @_shape.length == new_shape.length
        compatible_p &&= @_shape.each_with_index.all? {|x, i| x == 0 || x == new_shape[i] }
        unless compatible_p
          raise ArgumentError, "Expected shape #{new_shape} is incompatible with given shape #{@_shape}"
        end

        @_shape = new_shape
      end

      private def _check_and_get(arr_list, ctx)
        if arr_list
          return arr_list if ctx.equal?(Array)
          if ctx.nil?
            if arr_list.length == 1
              return arr_list[0]
            else
              ctx = Context.current
            end
          end
          ctx_list = @_ctx_map[ctx.device_type_id & 1]
          if ctx.device_id < ctx_list.length
            if (idx = ctx_list[ctx.device_id])
              return arr_list[idx]
            end
          end
          raise RuntimeError,
                "Parameter '#{@name}' was not initialized on context #{ctx}. " +
                "It was only initialized on #{@_ctx_list}."
        end
        unless @_deferred_init.empty?
          raise DeferredInitializationError,
                "Parameter '#{@name}' has not been initialized yet because " +
                "initialization was deferred.  Actual initialization happens " +
                "during the first forward pass.  " +
                "Please pass one batch of data through the network before " +
                "accessing Parameters.  " +
                "You can also avoid deferred initialization by specifying " +
                "in_units, num_features, etc., for network layers."
        end
        raise "Parameter '#{@name}' has not been initialized.  " +
              "Note that you should initialize parameters and " +
              "create Trainer with Block.collect_params instead " +
              "of Block.params because the later does not include " +
              "Parameters of nested child Blocks"
      end

      # (Re)initializes by loading from data.
      private def _load_init(data, ctx)
        if self.shape
          self.shape.each_with_index do |self_dim, i|
            if self_dim != 0 && self_dim != data.shape[i]
              raise "Failed loading Parameter '#{name}' from saved params: " +
                    "shape incompatible expected #{self.shape} vs saved #{data.shape}"
            end
          end
          self.shape.map!.with_index do |self_dim, i|
            self_dim != 0 ? self_dim : data.shape[i]
          end
        end
        if self.dtype
          data_dtype = data.dtype
          if self.dtype != data_dtype
            raise "Failed loading Parameter '#{name}' from saved params: " +
                  "dtype incompatible expected #{self.dtype} vs saved #{data_dtype}"
          end
        end
        ctx = [ctx] if ctx.is_a? MXNet::Context
        if @_data.nil?
          if @_deferred_init[1]
            if ctx && Set[*ctx] != Set[*@_deferred_init[1]]
              raise "Failed to load Parameter '#{name}' on #{ctx} because " +
                    "it was previous initialized on #{list_ctx}."
            end
            ctx = @_deferred_init[1]
          elsif ctx.nil?
            ctx = [MXNet.cpu]
          end
          _init_impl(data, ctx)
        else
          if ctx && Set[*ctx] != Set[*list_ctx]
            raise "Failed to load Parameter '#{name}' on #{ctx} because " +
                  "it was previous initialized on #{list_ctx}."
          end
          self.data = data
        end
        @_deferred_init = []
      end

      private def _finish_deferred_init
        return if @_deferred_init.empty?

        initializer, ctx, default_initializer, data = *@_deferred_init
        @_deferred_init = []
        unless self.shape && self.shape.inject(:*) > 0
          raise "Cannot initialize Parameter '#{@name}' because it has " +
                "invalid shape: #{self.shape}.  Please specify in_units, " +
                "in_channels, etc. for `Block`s."
        end

        MXNet::Autograd.pause do
          if data.nil?
            data = MXNet::NDArray.zeros(self.shape, dtype: self.dtype, ctx: MXNet.cpu)
            MXNet::Init.registry_manager.create(default_initializer).call(
              MXNet::Init::InitDesc.new(@name, {__init__: initializer}),
              data
            )
          end
          _init_impl(data, ctx)
        end
      end

      private def _init_impl(data, ctx_list)
        @_ctx_list = Array(ctx_list)
        @_ctx_map = [[], []]
        @_ctx_list.each_with_index do |ctx, i|
          dev_list = @_ctx_map[ctx.device_type_id & 1]
          while dev_list.length <= ctx.device_id
            dev_list << nil
          end
          dev_list[ctx.device_id] = i
        end

        @_data = @_ctx_list.map {|ctx| data.copy_to(ctx) }
        _init_grad
      end

      # Initialize grad buffers.
      private def _init_grad
        if grad_req == :null
          @_grad = nil
          return
        end
        @_grad = @_data.map {|d| MXNet::NDArray.zeros_like(d) }
        MXNet::Autograd.mark_variables(list_data, list_grad, grad_reqs: grad_req)
      end

      # Reduce data from multiple contexts.
      private def _reduce
        block = list_data
        MXNet::NDArray.add_n(*block.map {|w| w.copy_to(MXNet.cpu) }) / block.length
      end

      # Initializes parameter and gradient arrays.  Only used for `NDArray` API.
      #
      # ====Parameters
      #
      # +ctx+::          (Context or array of Contexts)
      #                  Desired contexts. Initialize Parameter on
      #                  given contexts.
      # +default_init+:: (Initializer, default MXNet::Uniform)
      #                  Default initializer.
      # +force_reinit+:: (boolean, default false)
      #                  Whether to force re-initialization if parameter
      #                  is already initialized.
      def init(initializer: nil, ctx: nil,
               default_initializer: MXNet::Init::Uniform.new,
               force_reinit: false)
        if @_data && !force_reinit
          warn "Parameter '#{@name}' is already initialized, ignoreing. " +
               "Set force_reinit: true to re-initialize."
          return
        end
        @_data = @_grad = nil

        ctx = Array(ctx || MXNet::Context.current)
        initializer ||= @initializer || default_initializer
        if self.shape.nil? || self.shape.inject(:*) <= 0
          if @_allow_deferred_init
            @_deferred_init = [initializer, ctx, default_initializer, nil]
            return
          end
          raise "Cannot initialize Parameter '#{@name}' because it has " +
                "invalid shape: #{self.shape}."
        end

        @_deferred_init = [initializer, ctx, default_initializer, nil]
        _finish_deferred_init
      end

      # Re-assign Parameter to other contexts.
      def reset_ctx(ctx)
        ctx = Array(ctx || MXNet::Context.current)
        if @_data
          data = _reduce()
          Autograd.pause do
            _init_impl(data, ctx)
          end
        elsif @_deferred_init
          @_deferred_init[1] = ctx
        else
          raise ArgumentError,
                "Cannot reset context for Parameter '#{@name}' because it " +
                "has not been initialized."
        end
      end

      # Sets this parameter's value on all contexts.
      def data=(data)
        # TODO:
      end

      # Returns a copy of this parameter on one context. Must have been
      # initialized on this context before.
      #
      # ====Parameters
      #
      # +ctx+:: (Context) Desired context.
      #
      # ====Returns
      #
      # NDArray on context.
      #
      def data(ctx: nil)
        _check_and_get(@_data, ctx)
      end

      # Returns copies of this parameter on all contexts, in the same
      # order as creation.
      #
      # ====Returns
      #
      # List of NDArrays.
      def list_data
        _check_and_get(@_data, Array)
      end

      # Returns a gradient buffer for this parameter on one context.
      # Must have been initialized on this context before.
      #
      # ====Parameters
      #
      # +ctx+:: (Context) Desired context.
      #
      # ====Returns
      #
      # NDArray on context.
      def grad(ctx: nil)
        if @_data && @_grad.nil?
          raise "Cannot get gradient array for Parameter '#{@name}' " +
                "because grad_req is null"
        end
        _check_and_get(@_grad, ctx)
      end

      # Returns gradient buffers on all contexts, in the same order as
      # creation.
      #
      # ====Returns
      #
      # List of NDArrays.
      def list_grad
        if @_data && @_grad.nil?
          raise "Cannot get gradient array for Parameter '#{@name}' " +
                "because grad_req is null"
        end
        _check_and_get(@_grad, Array)
      end

      # Returns a list of contexts this parameter is initialized on.
      def list_ctx
        if @_data.nil?
          return @_deferred_init[1] unless @_deferred_init.empty?
          raise "Parameter '#{@name}' has not been initialized"
        end
        @_ctx_list
      end

      def zero_grad
        return if @_grad.nil?
        @_grad.each {|i| i[0..-1] = 0 }
      end

      # Returns a symbol representing this parameter.
      def var
        @_var ||= MXNet::Symbol.var(@name, shape: shape, dtype: dtype,
                                    lr_mult: lr_mult, wd_mult: wd_mult,
                                    init: @initializer) # FIXME: init -> initializer
      end

      def cast(dtype)
        self.dtype = dtype
        return if @_data.nil?
        MXNet::Autograd.pause do
          @_data = @_data.map {|i| i.as_type(dtype) }
          return if @_grad.nil?
          @_grad = @_grad.map {|i| i.as_type(dtype) }
          MXNet::Autograd.mark_variables(@_data, @_grad, grad_reqs: grad_req)
        end
      end
    end

    # A constant parameter for holding immutable tensors.
    # `Constant`s are ignored by `Autograd` and `Trainer`, thus their values
    # will not change during training.  But you can still update their values
    # manually with the `data=` method.
    class Constant < Parameter
      class Init < MXNet::Init::Initializer
        def initialize(value, **kwargs)
          super(**kwargs)
          @value = value
        end

        private def init_weight(_, arr)
          @value.copy_to(arr)
        end
      end

      def initialize(name, value)
        unless value.is_a?(MXNet::NDArray)
          value = MXNet::NDArray.array(value)
        end
        @value = value
        initializer = Init.new(value)
        super(name, grad_req: :null, shape: value.shape, dtype: value.dtype, initializer: initializer)
      end
    end

    # A dictionary managing a set of parameters.
    #
    # ====Parameters
    #
    # +prefix+:: (string, default "") The prefix to be prepended to all
    #            Parameters' names created by this dict.
    # +shared+:: (ParameterDict or +nil+) If not +nil+, when this dict's
    #            #get method creates a new parameter, will first try to
    #            retrieve it from "shared" dict. Usually used for
    #            sharing parameters with another Block.
    #
    class ParameterDict < Hash
      def initialize(prefix_=nil, prefix: '', shared: nil)
        super()
        @prefix = prefix_ || prefix
        @shared = shared
      end

      # Prefix of this dict. It will be prepended to a Parameter's name
      # created with #get.
      attr_reader :prefix

      attr_reader :shared

      def inspect
        s = "%{name}(\n%{content}\n)"
        name = "#{@prefix}#{@prefix && ' '}"
        content = each_value.map {|v| MXNet::Utils.indent("  #{v}", 2) }.join("\n")
        s % {name: name, content: content}
      end

      alias to_s inspect

      # Retrieves a `Parameter` with name `@prefix + name`.  If not found, `get`
      # will first try to retrieve it from `shared` hash.  If still not found,
      # `get` will create a new `Parameter` with keyword arguments and insert
      # it to self.
      #
      # ====Parameters
      #
      # +name+:: (string) Name of the desired Parameter. It will be
      #          prepended with this dict's prefix.
      #
      # ====Returns
      #
      # The created or retrieved Parameter.
      #
      def get(name, **kwargs)
        name = name.to_s if name.is_a? ::Symbol
        name = @prefix + name.to_str
        param = get_impl(name)
        if param.nil?
          param = Parameter.new(name, **kwargs)
          self[name] = param
        else
          kwargs.each do |k, v|
            if param.has_attr?(k) && param[k]
              if v && v != param[k]
                raise ArgumentError,
                  "Cannot retrieve Parameter #{name} because desired " +
                  "attribute does not match with stored for attribute " +
                  "#{k}: desired #{v} vs stored #{param[k]}"
              end
            else
              param[k] = v
            end
          end
        end
        param
      end

      private def get_impl(name)
        return self[name] if self.has_key?(name)
        if @shared && @shared.has_key?(name)
          self[name] = @shared[name]
          return @shared[name]
        end
        nil
      end

      # Retrieves a `Constant` with name `prefix+name`.  If not found,
      # `get` will first try to retrieve it from *shared* dict.
      # If still not found, `get` will create a new `Constant` with keyword
      # arguments and insert it to self.
      def get_constant(name, value=nil)
        name = "#{prefix}#{name}"
        param = get_impl(name)
        if param
          if value
            unless param.is_a? Constant
              raise ArgumentError,
                    "Parameter '#{name}' already exists but it is not a constant."
            end
            value = value.to_narray if value.is_a? NDArray
            if param.shape != value.shape || param.value.to_narray != value
              raise ArgumentError,
                    "Constant '#{name}' already exists but it its value " +
                    "doesn't match new value."
            end
          end
        else
          unless value
            raise ArgumentError,
                  "No constant named '#{name}'.  Please specify value " +
                  "if you want to create a new constant."
          end
          param = Constant.new(name, value)
          self[name] = param
        end
        param
      end

      # Copies all Parameters in `other` to self.
      def update(other)
        other.each do |k, v|
          if has_key?(k)
            unless self[k].equal?(v)
              raise ArgumentError,
                "Cannot update self with other because they have different " +
                "Parameters with the same name #{k}"
            end
          end
        end
        super(other)
      end

      # Initializes all Parameters managed by this dictionary to be used for `NDArray` API.
      # It has no effect when using `Symbol` API.
      #
      # +ctx+:: (Context or array of Context)
      #         Desired contexts. Initialize Parameter on
      #         given contexts.
      # +force_reinit+:: (boolean, default false)
      #                  Whether to force re-initialization if parameters
      #                  are already initialized.
      def init(initializer: nil, ctx: nil, verbose: false, force_reinit: false)
        initializer ||= MXNet::Init::Uniform.new
        initializer.set_verbosity(verbose) if verbose
        each do |_, v|
          v.init(initializer: nil, ctx: ctx, default_initializer: initializer, force_reinit: force_reinit)
        end
      end

      # Sets all Parameters' gradient buffer to 0.
      def zero_grad
        each_value(&:zero_grad)
      end

      # Re-assign all Parameters to other contexts
      def reset_ctx(ctx)
        each_value {|x| x.reset_ctx(ctx) }
      end

      # Set an attribute to a new value for all Parameters.
      def set_attr(name, value)
        each_value {|x| x[name] = value }
      end

      # Save parameters to a file.
      def save(filename, strip_prefix='')
        args = {}
        each_value do |param|
          weight = param.send :_reduce
          if !param.name.start_with?(strip_prefix)
            raise ArgumentError,
                  "Prefix '#{strip_prefix}' is to be stripped before saving, " +
                  "but Parameter '#{param.name}' does not start with " +
                  "'#{strip_prefix}'.  If you are using Block.save_params, " +
                  "This may be due to your Block shares parameters from " +
                  "other Blocks or you forgot to use `with_name_scope` "
                  "method during init.  Consider switching to " +
                  "Block.collect_params.save and " +
                  "Block.collect_params.load instead."
          end
          args[param.name[strip_prefix.length .. -1]] = weight
        end
        MXNet::NDArray.save(filename, args)
      end

      # Load parameters
      def load(filename, ctx, allow_missing: false,
               ignore_extra: false, restore_prefix: '')
        unless restore_prefix.empty?
          each_keys do |name|
            unless name.start_with?(restore_prefix)
              raise ArgumentError,
                    "restore_prefix is '#{restore_prefix}' but Parameter " +
                    "name '#{name}' does not start with '#{restore_prefix}'."
            end
          end
        end
        lprefix = restore_prefix.length
        args = MXNet::NDArray.load(filename).transform_keys! do |key|
          restore_prefix + key.sub(/^(?:arg|aux):/, '')
        end
        unless allow_missing
          each_key do |name|
            unless args.has_key?(name)
              raise "Parameter '#{name[lprefix..-1]}' is missing in file " +
                    "'#{filename}', which contains parameters: " +
                    "#{brief_print_list(args.keys)}"
            end
          end
        end
        args.each do |name, value|
          if !has_key?(name) && !ignore_extra
            raise "Parameter '#{name[lprefix..-1]}' loaded from file " +
                  "'#{filename}' is not present in ParameterDict, " +
                  "choices are: #{brief_print_list(keys)}. " +
                  "Set ignore_extra to true to ignore. " +
                  "Please make sure source and target networks " +
                  "have the same prefix."
          end
          self[name].send :_load_init, value, ctx
        end
      end

      # Print at most `limit` elements of list.
      private def brief_print_list(ary, limit: 7)
        if ary.length > limit
          "#{brief_print_list(ary[0...limit.div(2)], limit: limit)}, ..., " +
          "#{brief_print_list(ary[(-limit.div(2) + 1)..-1])}"
        else
          ary.map(&:to_s).join(', ')
        end
      end
    end
  end
end
