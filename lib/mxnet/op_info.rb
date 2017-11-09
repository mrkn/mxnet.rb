module MXNet
  OpInfo = Struct.new(:name, :description, :args, :key_var_num_args, :return_type)
  OpArgInfo = Struct.new(:name, :type_info, :description)
end
