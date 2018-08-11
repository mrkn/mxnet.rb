require 'mxnet/gluon'

module MXNet::Gluon
  ##
  # Error for unfinished deferred initializations.
  #
  DeferredInitializationError = Class.new(RuntimeError)

  ##
  # A Container holding parameters (weights) of Blocks.
  #
  # Parameter holds a copy of the parameter on each Context after it
  # is initialized with #init. If +grad_req+ is not `:null`, it also
  # holds a gradient array on each Context.
  #
  class Parameter
    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +name+::                (string)
    #                         Name of this parameter.
    # +shape+::               (integer or array of integers)
    #                         Shape of this parameter.  By default,
    #                         shape is not specified.
    # +dtype+::               (symbol or string, default +:float32+)
    #                         Data type of this parameter.
    # +init+::                (Initializer, default +nil+)
    #                         The initializer to use.
    # +allow_deferred_init+:: (boolean, default +false+)
    #                         Is deferred initialization allowed.
    # +grad_req+::            (+:write+|+:add+|+:null+, default: +:write+)
    #                         Specifies how to update gradients.
    #                         - +:write+ means update is written to
    #                           the gradient.
    #                         - +:add+ means update is added to the
    #                           gradient. Call #zero_grad to clear the
    #                           gradient when using this option.
    #                         - +:null+ means a gradient is not
    #                           supported on this parameter.
    #
    def initialize(name, shape: nil, dtype: :float32, init: nil,
                   allow_deferred_init: false,
                   grad_req: :write)
      @var = nil
      @name = name
      shape = [shape] if shape.is_a?(Integer)
      self.shape = shape
      self.dtype = dtype
      @init = init
      @data = nil
      @grad = nil
      @deferred_init = []
      @allow_deferred_init = allow_deferred_init
      @grad_req = grad_req
      @trainer = nil
    end

    def name
      @name
    end

    def shape
      @shape
    end

    def shape=(shape)
      if @shape.nil?
        @shape = shape
      else
        unless @shape.length == shape.length &&
               @shape.zip(shape).all? { |i, j| i == j || i == 0 }
          raise RuntimeError,
                "Expected shape #{shape} is incompatible " \
                "with given shape #{@shape}."
        end
        @shape = shape
      end
    end

    def dtype
      @dtype
    end

    def dtype=(dtype)
      @dtype = dtype.is_a?(::String) || dtype.is_a?(::Symbol) ?
                 MXNet::DType.name2id(dtype) :
                 dtype
    end

    def trainer
      @trainer
    end

    def trainer=(trainer)
      @trainer = trainer
    end

    ##
    # Initializes parameter and gradient arrays. Only used for NDArray
    # API.
    #
    # ====Parameters
    #
    # +init+::         (Initializer, default +nil+)
    #                  The initializer to use. Overrides both `init`,
    #                  set when this instance was created, and
    #                  `default_init` in this call.
    # +ctx+::          (Context or array of Contexts, default +nil+)
    #                  Desired contexts. Initialize Parameter on given
    #                  contexts. A copy will be made for each
    #                  context. Note: copies are independent
    #                  arrays. Programmer is responsible for keeping
    #                  values consistent when updating. Normally
    #                  Trainer does this for you.
    # +default_init+:: (Initializer, default `:uniform`)
    #                  Default initializer.
    # +force_reinit+:: (boolean, default false)
    #                  Whether to force re-initialization if parameter
    #                  is already initialized.
    #
    # ====Examples
    #
    #     weight = MXNet::Gluon::Parameter.new('weight', shape: [2, 2])
    #     weight.init(ctx: MXNet.cpu)
    #     weight.data # => [[0.0068339, 0.0129982],...
    #     weight.grad # => [[0, 0],...
    #
    def init(init: nil, ctx: nil, default_init: :uniform, force_reinit: false)
      unless @data.nil? || force_reinit
        return
      end
      init = (@init || default_init) if init.nil?
      ctx = [MXNet.current_context] if ctx.nil?
      ctx = [ctx] if ctx.is_a?(MXNet::Context)
      @ctx = ctx
      @data = @grad = nil
      if @shape.nil? || @shape.flatten.inject(&:*) <= 0
        unless @allow_deferred_init
          raise RuntimeError,
                "Cannot initialize Parameter '#{@name}' because it has " \
                "invalid shape: #{@shape}."
        end
        @deferred_init = [ctx, init, nil]
        return
      end
      @deferred_init = [ctx, init, nil]
      finish_deferred_init
    end

    ##
    # Returns a list of contexts this parameter is initialized on.
    #
    def list_ctx
      if @data.nil?
        unless @deferred_init.empty?
          @deferred_init[0]
        else
          raise RuntimeError, "Parameter '#{@name}' has not been initialized."
        end
      else
        @ctx
      end
    end

    ##
    # Returns a symbol representing this parameter.
    #
    def var
      @var ||= MXNet::Symbol.var(@name, shape: @shape, dtype: @dtype)
    end

    ##
    # Returns a copy of this parameter on one context. Must have been
    # initialized on this context before.
    #
    # ====Parameters
    #
    # +ctx+:: (Context)
    #         Desired context.
    #
    # ====Returns
    #
    # NDArray on context.
    #
    def data(ctx: nil)
      check_and_get(@data, ctx)
    end

    ##
    # Returns copies of this parameter on all contexts, in the same
    # order as creation.
    #
    # ====Returns
    #
    # List of NDArrays.
    #
    def list_data
      check_and_get(@data, :all)
    end

    ##
    # Sets this parameter's value on all contexts.
    #
    def set_data(data)
      @shape = data.shape
      if @data
        check_and_get(@data, :all).each do |arr|
          arr[0..-1] = data
        end
      else
        unless @deferred_init.empty?
          @deferred_init[2] = data
        else
          raise RuntimeError, "Parameter '#{@name}' has not been initialized."
        end
      end
    end

    ##
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
    #
    def grad(ctx: nil)
      if !@grad && @data
        raise RuntimeError,
              "Cannot get gradient buffer for Parameter '#{name}' " \
              "because grad_req=:null."
      end
      check_and_get(@grad, ctx)
    end

    ##
    # Returns gradient buffers on all contexts, in the same order as
    # creation.
    #
    # ====Returns
    #
    # List of NDArrays.
    #
    def list_grad
      if !@grad && @data
        raise RuntimeError,
              "Cannot get gradient buffers for Parameter '#{name}' " \
              "because grad_req=:null."
      end
      check_and_get(@grad, :all)
    end

    ##
    # Sets gradient buffer to zero on all contexts.
    #
    def zero_grad
      if @grad
        check_and_get(@grad, :all).each do |arr|
          arr[0..-1] = 0
        end
      else
        if @deferred_init.empty?
          raise RuntimeError, "Parameter '#{@name}' has not been initialized."
        end
      end
    end

    def to_s
      "Parameter #{@name} (shape=#{@shape.inspect}, dtype=#{MXNet::DType.id2name(@dtype)})"
    end

    def ==(other)
      self.name == other.name && self.shape == other.shape
    rescue NoMethodError
      false
    end

    private

    ##
    # Reduce data from multiple contexts to cpu.
    #
    def reduce
      data = list_data.map { |d| d.copy_to(MXNet.cpu) }
      MXNet::NDArray.add_n(*data) / data.length
    end

    def check_and_get(arr_list, ctx)
      unless arr_list.nil?
        if ctx == :all
          return arr_list
        elsif ctx.nil?
          if arr_list.length == 1
            return arr_list[0]
          end
          ctx = MXNet.current_context
        end
        if data = arr_list.find { |arr| arr.context == ctx }
          return data
        end
        raise RuntimeError,
              "Parameter '#{@name}' was not initialized on context #{ctx}."
      end
      unless @deferred_init.empty?
        raise DeferredInitializationError,
              "Parameter '#{@name}' has not been initialized yet because " \
              "initialization was deferred. Actual initialization happens " \
              "during the first forward pass. Please pass one batch of " \
              "data through the network before accessing Parameters."
      end
      raise RuntimeError,
            "Parameter '#{@name}' has not been initialized. You should " \
            "initialize parameters and create a Trainer with #collect_params " \
            "instead of #params because the later does not include Parameters " \
            "of nested child Blocks."
    end

    def load_and_init(ctx, data)
      ctx = ctx.is_a?(MXNet::Context) ? [ctx] : ctx
      if @data
        set_data(data)
      else
        init_impl(ctx, data)
      end
    end

    def finish_deferred_init
      return if @deferred_init.empty?
      ctx, default_init, data = @deferred_init
      @deferred_init = []
      if @shape.nil? || @shape.flatten.inject(&:*) <= 0
        raise RuntimeError,
              "Cannot initialize Parameter '#{@name}' because it has " \
              "invalid shape: #{@shape}."
      end
      MXNet::Autograd.pause do
        unless data
          data = MXNet::NDArray.zeros(@shape, dtype: @dtype, ctx: MXNet.cpu)
          MXNet::Initializer.create(default_init)[data]
        end
        init_impl(ctx, data)
      end
    end

    def init_impl(ctx, data)
      @data = ctx.map { |c| data.copy_to(c) }
      grad = MXNet::NDArray.zeros(@shape, dtype: @dtype, ctx: MXNet.cpu)
      init_grad(ctx, grad)
    end

    def init_grad(ctx, grad)
      if @grad_req == :null
        @grad = nil
        return
      end
      @grad = ctx.map { |c| grad.copy_to(c) }
      MXNet::Autograd.mark_variables(
        check_and_get(@data, :all),
        check_and_get(@grad, :all),
        grad_reqs: @grad_req
      )
    end
  end

  ##
  # A dictionary managing a set of Parameters.
  #
  class ParameterDict
    include Enumerable

    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +prefix+:: (string, default "")
    #            The prefix to be prepended to all Parameters' names
    #            created by this dict.
    # +shared+:: (ParameterDict or +nil+)
    #            If not +nil+, when this dict's #get method creates a
    #            new parameter, will first try to retrieve it from
    #            "shared" dict. Usually used for sharing parameters
    #            with another Block.
    #
    def initialize(prefix: '', shared: nil)
      @params = {}
      @prefix = prefix
      @shared = shared
    end

    def each(&block)
      @params.each(&block)
    end

    def keys
      @params.keys
    end

    def values
      @params.values
    end

    ##
    # Prefix of this dict. It will be prepended to a Parameter's name
    # created with #get.
    #
    def prefix
      @prefix
    end

    ##
    # Retrieves a Parameter with name "<prefix><name>". If not found,
    # #get will first try to retrieve it from "shared" dict. If still
    # not found, #get will create a new Parameter with key-word
    # arguments and both store and return it.
    #
    # ====Parameters
    #
    # +name+:: (string)
    #          Name of the desired Parameter. It will be prepended
    #          with this dict's prefix.
    #
    # ====Returns
    #
    # The created or retrieved Parameter.
    #
    def get(name, **kwargs)
      name = @prefix + name
      unless param = _get(name.to_sym)
        param = @params[name.to_sym] = Parameter.new(name, **kwargs)
      else
        param
      end
    end

    ##
    # Copies all Parameters in +other+ to this dict.
    #
    def update(other)
      other.each do |k, v|
        if @params[k] && @params[k] != v
          raise ArgumentError, "Cannot update because keys have different values: #{@params[k]}, #{v}"
        end
      end
      other.each do |k, v|
        @params[k] = v
      end
    end

    ##
    # Initializes all Parameters managed by this dict to be used for
    # NDArray API. It has no effect when using Symbol API.
    #
    # ====Parameters
    #
    # +init+::         (Initializer, default +nil+)
    #                  The initializer to use.
    # +ctx+::          (Context or array of Context)
    #                  Desired contexts. Initialize Parameter on given
    #                  contexts.
    # +force_reinit+:: (boolean, default false)
    #                  Whether to force re-initialization if parameters
    #                  are already initialized.
    #
    def init(init: nil, ctx: nil, force_reinit: false)
      values.each { |v| v.init(init: init, ctx: ctx, force_reinit: force_reinit) }
    end

    def to_s
      "ParameterDict (\n" +
        self.inject('') { |a, (n, p)| a + "  #{p}\n" } +
        ')'
    end

    def ==(other)
      self.prefix == other.prefix && self.each.to_a == other.each.to_a
    rescue NoMethodError
      false
    end

    protected

    def _get(name)
      if value = @params[name]
        value
      elsif @shared && value = @shared._get(name)
        @params[name] = value
        value
      end
    end
  end
end
