module MXNet
  class NDArray
    include HandleWrapper

    def self.ones(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      dtype = Utils.dtype_id(dtype)
      Internal._ones(shape: shape, ctx: ctx, dtype: dtype, **kwargs)
    end

    def self.zeros(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      dtype = Utils.dtype_id(dtype)
      Internal._zeros(shape: shape, ctx: ctx, dtype: dtype, **kwargs)
    end

    def self.arange(start, stop=nil, step: 1.0, repeat: 1, ctx: nil, dtype: :float32)
      ctx ||= Context.default
      dtype = Utils.dtype_name(dtype)
      Internal._arange(start: start, stop: stop, step: step, repeat: repeat, dtype: dtype, ctx: ctx)
    end

    def inspect
      shape_info = shape.join('x')
      ary = to_narray.inspect.lines[1..-1].join
      "\n#{ary}\n<#{self.class} #{shape_info} @#{context}>"
    end

    def context
      dev_typeid, dev_id = _get_context_params
      Context.new(dev_typeid, dev_id)
    end

    # Returns a sliced view of this array.
    #
    # @param [Integer, Range, Array] key  Indexing key.
    # @return [NDArray] a sliced view of this array.
    def [](key)
      case key
      when Integer
        if key > shape[0] - 1
          raise IndexError, "index #{key} is out of bounds for axis 0 with size #{shape[0]}"
        end
        _at(key)
      when Range, Enumerator
        start, stop, step = MXNet::Utils.decompose_slice(key)
        if step && step != 1
          raise ArgumentError, 'slice step cannot be zero' if step == 0
          Ops.slice(self, begin: [start], end: [stop], step: [step])
        elsif start || stop
          _slice(start, stop)
        else
          self
        end
      when Array
        keys = key
        shape = self.shape
        unless shape.length >= keys.length
          raise IndexError, "Slicing dimensions exceeds array dimensions, #{keys.length} vs #{shape.length}"
        end
        out_shape, begins, ends = [], [], []
        keys.each_with_index do |key, idx|
          case key
          when Integer
            begins << key
            ends << key + 1
          when Range
            begins << (key.begin.nil? || key.begin == -Float::INFINITY) ? 0 : key.begin
            ends << (key.end.nil? || key.end == Float::INFINITY) ? shape[i] : key.end
            out_shape << ends.last - begins.last
          else
            raise IndexError, "NDArray does not support slicing with key #{key} of type #{key.class}"
          end
        end
        out_shape.concat(shape[keys.length..-1])
        out_shape << 1 if out_shape.empty?
        return slice(begins, ends).reshape(out_shape)
      else
        raise IndexError, "NDArray does not support slicing with key #{key} of type #{key.class}"
      end
    end

    def []=(key, value)
      view = self[key]
      case key
      when Numeric
        Internal._set_value(src: value.to_f, out: view)
      when Range, Enumerator
        start, stop, step = MXNet::Utils.decompose_slice(key)
        if step.nil? || step == 1
          unless start == 0 && (stop.nil? || stop == shape[0])
            sliced_arr = _slice(start, stop)
            sliced_arr[0..-1] = value
            return value
          end
          _fill_by(value)
          return value
        end
        # non-trivial step, use _slice_assign or _slice_assign_scalar
        raise NotImplementedError
      else
        raise NotImplementedError
      end
    end

    def _fill_by(value)
      case value
      when NDArray
        if __mxnet_handle__ != value.send(:__mxnet_handle__)
          Internal._copyto(value, out: self)
        end
      when Numeric
        Internal._full(shape: self.shape, ctx: self.context,
                       dtype: self.dtype, value: value.to_f, out: self)
      else
        case
        when value.is_a?(Array)
          raise NotImplementedError, "Array is not supported yet"
        when defined?(Numo::NArray) && value.is_a?(Numo::NArray)
          # require 'mxnet/narray_helper'
          # TODO: MXNet::NArrayHelper.sync_copyfrom(self, value)
          raise NotImplementedError, "NArray is not supported yet"
        when defined?(NMatrix) && value.is_a?(NMatrix)
          # require 'mxnet/mxnet_helper'
          # TODO: _sync_copyfrom_nmatrix(value)
          raise NotImplementedError, "NMatrix is not supported yet"
        when defined?(Vector) && value.is_a?(Vector)
          raise NotImplementedError, "Vector is not supported yet"
        when defined?(Matrix) && value.is_a?(Matrix)
          raise NotImplementedError, "Matrix is not supported yet"
        else
          raise TypeError, "NDArray does not support assignment with non-array-like " +
            "object #{value.to_s} of #{value.class} class"
        end
      end
    end
    private :_fill_by

    def ndim
      shape.length
    end
    alias rank ndim

    def size
      shape.inject(:*)
    end
    alias length size

    def transpose(axes: nil)
      Ops.transpose(self, axes: axes)
    end

    def as_scalar
      unless shape == [1]
        raise TypeError, "The current array is not a scalar"
      end
      to_a[0]
    end

    def +@
      self
    end

    def -@
      Internal._mul_scalar(self, scalar: -1.0)
    end

    def +(other)
      case other
      when NDArray
        Ops.broadcast_add(self, other)
      else
        super
      end
    end

    def -(other)
      case other
      when NDArray
        Ops.broadcast_sub(self, other)
      else
        super
      end
    end

    def *(other)
      case other
      when NDArray
        Ops.broadcast_mul(self, other)
      else
        super
      end
    end

    def /(other)
      case other
      when NDArray
        Ops.broadcast_div(self, other)
      else
        super
      end
    end

    def %(other)
      case other
      when NDArray
        Ops.broadcast_mod(self, other)
      else
        super
      end
    end

    def **(other)
      case other
      when NDArray
        Ops.broadcast_power(self, other)
      else
        super
      end
    end

    def ==(other)
      case other
      when NDArray
        Ops.broadcast_equal(self, other)
      else
        super
      end
    end

    def !=(other)
      case other
      when NDArray
        Ops.broadcast_not_equal(self, other)
      else
        super
      end
    end

    def >(other)
      case other
      when NDArray
        Ops.broadcast_greater(self, other)
      else
        super
      end
    end

    def >=(other)
      case other
      when NDArray
        Ops.broadcast_greater_equal(self, other)
      else
        super
      end
    end

    def <(other)
      case other
      when NDArray
        Ops.broadcast_lesser(self, other)
      else
        super
      end
    end

    def <=(other)
      case other
      when NDArray
        Ops.broadcast_lesser_equal(self, other)
      else
        super
      end
    end

    # Returns a Numo::NArray object with value copied from this array.
    def to_narray
      require 'mxnet/narray_helper'
      self.to_narray
    end

    module Ops
      def self._import_ndarray_operations
        LibMXNet._each_op_names do |op_name|
          op_handle = LibMXNet._get_op_handle(op_name)
          op_info = LibMXNet._get_op_info(op_handle)
        end
      end
    end
  end

  NDArray::CONVERTER = []

  def self.NDArray(array_like, ctx: nil, dtype: :float32)
    ctx ||= MXNet.current_context
    for type, converter in NDArray::CONVERTER
      if array_like.is_a?(type)
        if converter.respond_to? :to_ndarray
          return converter.to_ndarray(array_like, ctx: ctx, dtype: dtype)
        elsif converter.respond_to? :call
          return converter.call(array_like, ctx: ctx, dtype: dtype)
        end
      end
    end
    raise TypeError, "Unable convert #{array_like.class} to MXNet::NDArray"
  end
end
