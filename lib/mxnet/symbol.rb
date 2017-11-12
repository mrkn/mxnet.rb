module MXNet
  class Symbol
    include HandleWrapper

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

    def -@
      self * -1.0
    end

    def +(other)
      case other
      when Symbol
        Internal._Plus(self, other)
      else
        # TODO
      end
    end

    def -(other)
      case other
      when Symbol
        Internal._Minus(self, other)
      else
        # TODO
      end
    end

    def *(other)
      case other
      when Symbol
        Internal._Mul(self, other)
      else
        # TODO
      end
    end

    def /(other)
      case other
      when Symbol
        Internal._Div(self, other)
      else
        # TODO
      end
    end

    def %(other)
      case other
      when Symbol
        Internal._Mod(self, other)
      else
        # TODO
      end
    end

    def **(other)
      case other
      when Symbol
        Internal._Power(self, other)
      else
        # TODO
      end
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
