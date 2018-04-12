module MXNet
  class Symbol
    module OperationDelegator
      def self.define_delegator(mod, handle, op_info)
        dtype_name = nil
        ary_name = nil
        signature = []
        ndsignature = []
        ndarg_names = []
        kwarg_names = []
        op_info.args.each do |arg|
          name = arg.name
          case name.to_s
          when 'begin', 'end', /\A[A-Z]/
            name = :"_#{name}"
          end
          if name == :dtype
            dtype_name = name
            signature << "#{name}: nil"
          elsif arg.type_info.start_with?('NDArray') || arg.type_info.start_with?('Symbol')
            raise "Op can only have one argument with variable size and it must be the last argument." if ary_name
            if arg.type_info.end_with?('[]')
              ndsignature << "*#{name}"
              ary_name = name
            else
              ndsignature << "#{name}=nil"
              ndarg_names << name
            end
          else
            signature << "#{name}: nil"
            kwarg_names << name
          end
        end
        signature << 'name: nil'
        signature << 'attr: nil'
        signature << 'out: nil'
        signature << '**kwargs'
        signature = ndsignature + signature

        code = []
        if ary_name
          lineno = __LINE__ + 2
          code << <<-RUBY
  def #{op_info.func_name}(*#{ary_name}, **kwargs)
    #{ary_name}.each do |i|
      raise TypeError, "unexpected positional arguments \#{i.class} (expect NDArray)" unless i.kind_of? MXNet::Symbol
    end
    sym_args = #{ary_name}
          RUBY
          if dtype_name
            code << <<-RUBY
    if kwargs.has_key?(:#{dtype_name})
      kwargs[:#{dtype_name}] = MXNet::DType.name(#{dtype_name})
    end
            RUBY
          end
          code << <<-RUBY
    kwargs.delete_if { |k, v| v.nil? } # TODO: check if this is necessary or not
    attr = kwargs.delete(:attr)
    kwargs.update(AttrScope.current.get(attr))
    name = kwargs.delete(:name)
    name = Name::NameManager.current.get(name, :'#{op_info.func_name.to_s.downcase}')
    _ = kwargs.delete(:out)
    keys = []
    vals = []
    sym_kwargs = {}
    kwargs.each do |k, v|
      case v
      when MXNet::Symbol
        sym_kwargs[k] = v
      else
        keys << k
        vals << v
      end
    end
          RUBY
          if op_info.key_var_num_args
            code << <<-RUBY
    if kwargs.include?(:#{op_info.key_var_num_args})
      keys << :#{op_info.key_var_num_args}
      vals << sym_args.length + sym_kwargs.length
    end
            RUBY
          end
          code << <<-RUBY
    return LibMXNet.symbol_creator(#{handle}, sym_args, sym_kwargs, keys, vals, name)
          RUBY
        else # if ary_name
          lineno = __LINE__ + 2
          code << <<-RUBY
  def #{op_info.func_name}(#{signature.join(', ')})
    kwargs.delete_if { |k, v| v.nil? } # TODO: check if this is necessary or not
    kwargs.update(AttrScope.current.get(attr))
    sym_kwargs = {}
    keys = []
    vals = []
    kwargs.each do |k, v|
      case v
      when MXNet::Symbol
        sym_kwargs[k] = v
      else
        keys << k
        vals << v
      end
    end
          RUBY
          ndarg_names.each do |name|
            code << <<-RUBY
    if #{name}
      raise TypeError, "unexpected type of argument \#{#{name}.class} (expected #{MXNet::Symbol})" unless #{name}.kind_of? MXNet::Symbol
      sym_kwargs[:#{name}] = #{name}
    end
            RUBY
          end
          kwarg_names.each do |name|
            code << <<-RUBY
    if #{name}
      keys << :#{name}
      vals << #{name}
    end
            RUBY
          end
          if dtype_name
            code << <<-RUBY
    if #{dtype_name}
      keys << :#{dtype_name}
      vals << MXNet::DType.name(#{dtype_name})
    end
            RUBY
          end
        end # if ary_name

        code << <<-RUBY
    name = Name::NameManager.current.get(name, :'#{op_info.func_name.to_s.downcase}')
    return LibMXNet.symbol_creator(#{handle}, nil, sym_kwargs, keys, vals, name)
  end
  module_function :#{op_info.func_name}
        RUBY

        mod.module_eval(code.join(''), __FILE__, lineno)
      end
    end
  end
end
