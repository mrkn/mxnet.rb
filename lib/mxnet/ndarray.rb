module MXNet
  class NDArray
    module DType
      NAME_TO_ID = {
        float32: 0,
        float64: 1,
        float16: 2,
        uint8: 3,
        int32: 4,
        int8: 5,
        int64: 6,
      }.freeze

      ID_TO_NAME = NAME_TO_ID.inject({}) {|h, (k, v)| h.update(v => k) }.freeze

      NAME_TO_ID.each do |name, id|
        const_set(name.to_s.upcase, id)
      end

      def self.dtype_name(val)
        case val
        when Integer
          ID_TO_NAME[val]
        when Symbol
          ID_TO_NAME[val]
        else
          ID_TO_NAME[val.to_str.to_sym]
        end
      end
    end

    def self._dtype(name)
      DType.const_get(name.to_s.upcase)
    rescue NameError
      raise ArgumentError, 'unknown dtype name: #{name}'
    end

    def self.ones(shape, ctx=nil, dtype=DType::FLOAT32, **kwargs)
      ctx ||= Context.default
      Ops._ones(shape, ctx, dtype, **kwargs)
    end

    def self.zeros(shape, ctx=nil, dtype=DType::FLOAT32, **kwargs)
      ctx ||= Context.default
      Ops._zeros(shape, ctx, dtype, **kwargs)
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
