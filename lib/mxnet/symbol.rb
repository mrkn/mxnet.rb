module MXNet
  class Symbol
    include HandleWrapper

    # NATIVE: self.load
    # NATIVE: self.load_json

    def self.arange(start, stop=nil, step: 1.0, repeat: 1, name: nil, dtype: nil)
      dtype ||= :float32
      Internal._arange(start: start, stop: stop, step: step, repeat: repeat, name: name, dtype: dtype)
    end

    # Sets an attribute of the symbol.
    #
    # @param kwargs  The attributes to set
    def _set_attr(**kwargs)
      # TODO
      # kwargs.each do |k, v|
      #   raise ArgumentError, 'Set Attr only accepts string value' unless v.is_a?(String)
      # end
    end

    # def infer_type

    # Infers the shape of all arguments and all outputs given the known shapes of some arguments.
    #
    # This function takes the known shape of some arguments in either positional way or keyword argument way as input.
    # It returns a tuple of `nil` values if there is not enough information to deduce the missing shapes.
    #
    # Example:
    #
    #     > a = MXNet.var(:a)
    #     > b = MXNet.var(:b)
    #     > c = a + b
    #     > arg_shapes, out_shapes, aux_shapes = c.infer_shape(a: [3, 3])
    #     > arg_shapes
    #     [[3, 3], [3, 3])
    #     > out_shapes
    #     [[3, 3]]
    #     > aux_shapes
    #     []
    #     > c.infer_shape(a: [0, 3])  # 0s in shape means unknown dimension. So, returns nil.
    #     [nil, nil, nil]
    #
    # Inconsistencies in the known shapes will cause an error to be raised.
    # See the following example:
    #
    #     > data = MXNet.var(:data)
    #     > out = MXNet.FullyConnected(data: data, name: :fc1, num_hidden: 1000)
    #     > out = MXNet.Activation(data: out, act_type: :relu)
    #     > out = MXNet.FullyConnected(data: out, name: :fc2, num_hidden: 10)
    #     > weight_shape = [1, 100]
    #     > data_shape = [100, 100]
    #     > out.infer_shape(data: data_shape, fc1_weight: weight_shape)
    #     Error in operator fc1: Shape inconsistent, Provided=[1, 100], inferred shape=[1000, 100]
    #
    # @param *args  Shape of arguments in a positional way. Unknown shape can be marked as nil.
    # @param **kwargs  Keyword arguments of the knwon shapes.
    #
    # @return [Array<Array<Array, nil>>]  Three arrays of argument shapes.
    #   The first array is array of argument shapes.
    #   The second array is array of output shapes.
    #   The third array is array of auxiliary state shapes.
    #   The order of elements in each array is same as the order of `list_arguments`, `list_outputs`,
    #   and `list_auxiliary_states`, respectively.
    def infer_shape(*args, **kwargs)
      res = infer_shape_impl(false, *args, **kwargs)
      if res[1].nil?
        arg_shapes, _, _ = infer_shape_impl(true, *args, **kwargs)
        arg_names = list_arguments
        unknowns = []
        arg_names.lazy.zip(arg_shapes).each do |name, shape|
          if !shape # TODO: || !_numpy.prod(shape)
            if unknowns.length >= 10
              unknowns << '...'
              break
            end
            unknowns << "#{name}: #{shape}"
          end
        end
        warn <<-WARN
#{caller(1, 1).first}: Cannot decide shape for the following arguments (0s in shape means unknown dimensions). Consider providing them as input:
\t#{unknowns.join("\n\t")}
        WARN
      end
      return res
    rescue MXNet::Error
      puts "infer_shape error. Arguments:"
      args.each_with_index do |arg, i|
        puts "  ##{i}: #{arg}"
      end
      kwargs.each do |k, v|
        puts "  #{k}: #{v}"
      end
      raise
    end

    # TODO: documentation
    def infer_shape_partial(*args, **kwargs)
      infer_shape_impl(true, *args, **kwargs)
    end

    # NATIVE: save
    # NATIVE: to_json

    # Evaluates a symbol given argumens.
    # 
    # The `eval` method combines a call to `bind` (which returns an executer)
    # with a call to `forward` (executor method).
    # For the common use case, where you might repeatedly evaluate with same arguments,
    # eval is slow.
    # In that case, you should call `bind` once and then repeatedly call `forward`.
    # This function allows simpler syntax for less cumbersome introspection.
    # 
    # Example:
    # 
    #     > a = MXNet.var(:a)
    #     > b = MXNet.var(:b)
    #     > c = a + b
    #     > ex = c.eval(ctx: MXNet.cpu, a: MXNet::NDArray.ones([2, 3]), b: MXNet::NDArray.ones([2, 3]))
    #     [#<MXNet::NDArray 2x3 @cpu(0)>
    #     > ex[0].reshape([6]).to_a
    #     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    #
    # @param ctx  [MXNet::Context]  The device context the generated executor to run on.
    # @param kwargs  [Hash]  Input arguments to the symbol.  All the arguments must be provided.
    #
    # @return [Array<MXNet::NDArray>]  An array of NDArrays corresponding to
    #   the values taken by each symbol when evaluated on given args.
    #   When called on a single symbol (not a group), the result will be
    #   an array with one element.
    def eval(ctx: nil, **kwargs)
      ctx ||= Context.default_ctx
      bind(ctx, kwargs).forward
    end

    def +@
      self
    end

    # Numerical negative, element-wise.
    def -@
      self * -1.0
    end

    def +(other)
      case other
      when Symbol
        Internal._Plus(self, other)
      when Numeric
        Internal._PlusScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    def -(other)
      case other
      when Symbol
        Internal._Minus(self, other)
      when Numeric
        Internal._MinusScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    def *(other)
      case other
      when Symbol
        Internal._Mul(self, other)
      when Numeric
        Internal._MulScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    def /(other)
      case other
      when Symbol
        Internal._Div(self, other)
      when Numeric
        Internal._DivScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    def %(other)
      case other
      when Symbol
        Internal._Mod(self, other)
      when Numeric
        Internal._ModScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    def **(other)
      case other
      when Symbol
        Internal._Power(self, other)
      when Numeric
        Internal._PowerScalar(self, scalar: other)
      else
        raise TypeError, "An instance of #{other.class} class is not supported"
      end
    end

    class SwappedOperationAdapter < Struct.new(:scalar)
      def +(symbol)
        symbol + scalar
      end

      def -(symbol)
        Internal._RMinusScalar(symbol, scalar: scalar)
      end

      def *(symbol)
        symbol * scalar
      end

      def /(symbol)
        Internal._RDivScalar(symbol, scalar: scalar)
      end

      def %(symbol)
        Internal._RModScalar(symbol, scalar: scalar)
      end

      def **(symbol)
        raise NotImplementedError, "Method ** is not implemented for Symbol and only available in NDArray."
      end
    end

    def coerce(other)
      [SwappedOperationAdapter.new(other), self]
    end
  end

  Variable = Symbol # deprecated

  # Creates a symbolic variable with specified name.
  #
  # @param name
  # @param attr
  # @param shape
  # @param lr_mult
  # @param wd_mult
  # @param dtype
  # @param init
  # @param stype
  # @param kwarg  Additinoal attribute variables
  #
  # @return [Symbol]  A symbol corresponding to an input to the computation graph.
  def self.var(name, attr: nil, shape: nil, lr_mult: nil, wd_mult: nil, dtype: nil,
               init: nil, stype: nil, **kwargs)
    unless name.is_a?(String) || name.is_a?(::Symbol)
      raise TypeError, 'Expect a String or a Symbol for variable `name`'
    end
    handle = LibMXNet.create_variable(name)
    sym = MXNet::Symbol.new(handle)
    attr = AttrScope.current.get(attr)
    attr ||= {}
    attr[:__shape__] = shape.to_s if shape
    attr[:__lr_mult__] = lr_mult if lr_mult
    attr[:__wd_mult__] = wd_mult if wd_mult
    attr[:__dtype__] = MXNet::DType.name(dtype) if dtype
    if init
      init = init.to_json unless init.is_a?(String) || init.is_a?(::Symbol)
      attr[:__init__] = init
    end
    # attr[:__storage_type__] = str(_STORAGE_TYPE_STR_TO_ID[stype] if stype
    kwargs.each do |k, v|
      if k.start_with?('__') && k.end_with?('__')
        attr[k] = v
      else
        raise ArgumentError, "Attribute name=#{k} is not supported." +
          ' Additional attributes must start and end with double underscores,' +
          ' e.g., __yourattr__'
      end
    end
    sym._set_attr(**attr)
    return sym
  end
end
