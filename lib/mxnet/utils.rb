module MXNet
  module Utils
    module_function

    def dtype_id(name_or_id)
      case name_or_id
      when String, ::Symbol
        MXNet::DType.name2id(name_or_id)
      when Integer
        name_or_id
      else
        raise TypeError, "wrong type of dtype: #{name_or_id.class} (expected Symbol or Integer)"
      end
    end

    def dtype_name(name_or_id)
      case name_or_id
      when String, ::Symbol
        name_or_id.to_s
      when Integer
        MXNet::DType.id2name(name_or_id)
      else
        raise TypeError, "wrong type of dtype: #{name_or_id.class} (expected Symbol or Integer)"
      end
    end
  end
end
