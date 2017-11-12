module MXNet
  class NDArray
    include HandleWrapper

    def self.ones(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      dtype = Utils.dtype_id(dtype)
      Internal._ones(shape, ctx, dtype, **kwargs)
    end

    def self.zeros(shape, ctx=nil, dtype=:float32, **kwargs)
      ctx ||= Context.default
      dtype = Utils.dtype_id(dtype)
      Internal._zeros(shape, ctx, dtype, **kwargs)
    end

    def self.arange(start, stop=nil, step: 1.0, repeat: 1, ctx: nil, dtype: :float32)
      ctx ||= Context.default
      dtype = Utils.dtype_name(dtype)
      Internal._arange(start: start, stop: stop, step: step, repeat: repeat, dtype: dtype, ctx: ctx)
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
      when Range
        if key.begin || key.end
          _slice(key.begin, key.end)
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

    def ndim
      shape.length
    end
    alias rank ndim

    def size
      shape.inject(:*)
    end
    alias length size

    def transpose(axes=nil)
      Ops.transpose(self, axes)
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
      Internal._mul_scalar(self, -1.0)
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

    module Ops
      def self._import_ndarray_operations
        LibMXNet._each_op_names do |op_name|
          op_handle = LibMXNet._get_op_handle(op_name)
          op_info = LibMXNet._get_op_info(op_handle)
        end
      end
    end
  end
end
