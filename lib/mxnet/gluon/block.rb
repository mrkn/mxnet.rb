require 'mxnet/gluon'

module MXNet
  module Gluon
    # Scope for collecting child Blocks.
    class BlockScope
      TLS_KEY = :"#{self.name}.current"
      private_constant :TLS_KEY

      def self.current
        Thread.current[TLS_KEY]
      end

      def self.current=(block_scope)
        Thread.current[TLS_KEY] = block_scope
      end

      # Create prefix and params for new `Block`.
      def self.make_prefix_and_params(prefix, params, hint)
        unless self.current
          prefix ||= MXNet::Name::NameManager.current.get(nil, hint) + '_'
          if params
            params = ParameterDict.new(params.prefix, shared: params)
          else
            params = ParameterDict.new(prefix)
          end
          return [prefix, params]
        end

        unless prefix
          count = self.current.counter[hint] || 0
          prefix = "#{hint}#{count}_"
          self.current.counter[hint] = count + 1
        end
        if params
          params = ParameterDict.new(params.prefix, shared: params)
        else
          parent = self.current.block.params
          params = ParameterDict.new(parent.prefix + prefix, shared: parent.shared)
        end
        [self.current.block.prefix + prefix, params]
      end

      def initialize(block)
        @block = block
        @counter = {}
        @old_scope = nil
        @name_scope = nil
      end

      attr_reader :block, :counter

      def enter
        return _enter unless block_given?
        begin
          _enter
          yield
        ensure
          exit
        end
      end

      private def _enter
        return self if @block.empty_prefix?
        @old_scope = BlockScope.current
        BlockScope.current = self
        @name_scope = MXNet::Name::Prefix.new(prefix: @block.prefix)
        @name_scope.enter
        self
      end

      def exit
        return if @block.empty_prefix?
        @name_scope.exit
        @name_scope = nil
        BlockScope.current = @old_scope
        nil
      end
    end

    # Base class for all neural network layers and models.  Your models should
    # be subclasses of this class.
    #
    # ...
    class Block
      def initialize(prefix: nil, params: nil)
        @empty_prefix = (prefix == '')
        @prefix, @params = BlockScope.make_prefix_and_params(prefix, params, alias_name)
        @name = @prefix.end_with?('_') ? @prefix[0...-1] : @prefix
        @scope = BlockScope.new(self)
        @children = {}
        @reg_params = {}
        @attributes = {}
      end

      def inspect
        s = "%{name}(\n%{modstr}\n)"
        modstr = @attributes.map { |k, v|
          '  (%{key}): %{block}' % {
            key: k,
            block: Utils.indent(v.inspect, 2)
          } if v.is_a?(Block)
        }.compact.join("\n")
        s % {name: self.class.name, modstr: modstr}
      end

      def __getattr__(name)
        @attributes[name]
      end

      alias_method :[], :__getattr__

      def has_attr?(name)
        @attributes.has_key?(name)
      end

      # Registers parameters.
      def __setattr__(name, value)
        if has_attr?(name)
          existing = __getattr__(name)
          if (existing.is_a?(Parameter) || existing.is_a?(Block)) && !value.is_a?(existing.class)
            raise TypeError,
              "Changing attribute type for %{name} from %{type1} to %{type2} is not allowed." % {
                name: name, type1: existing.class, type2: value.class }
          end
        end

        case value
        when Block
          register_child(value, name)
        when Parameter
          if @reg_params.include?(name)
            raise ArgumentError,
              "Overriding Parameter attribute #{name} is not allowd. " +
              "If you want to share parameters between blocks, please set " +
              "'params' at Block construction instead."
          end
          @reg_params[name] = value
        end

        @attributes[name] = value
      end

      alias_method :[]=, :__setattr__

      private def alias_name
        self.class.name[/:?(\w+)\z/, 1].downcase
      end

      # Prefix of this `Block`.
      attr_reader :prefix

      def empty_prefix?
        @empty_prefix
      end

      # Name of this `Block`, without `'_'` in the end.
      attr_reader :name

      # Return a name scope object managing a child `Block` and parameter names.
      attr_reader :scope

      def with_name_scope(&block)
        scope.enter &block
      end

      # Returns this Block's ParameterDict (does not include its
      # children's parameters).
      attr_reader :params

      # Returns a ParameterDict containing this Block's and all of its
      # children's Parameters. Also can return the Parameters that match
      # some given regular expressions.
      #
      # For example, collect the specified Parameters for
      # 'conv1_weight', 'conv1_bias', 'fc_weight' and 'fc_bias':
      #
      #     model.collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')
      #
      # or, alternatively, collect all parameters whose names end with
      # 'weight' or 'bias':
      #
      #     model.collect_params('.*weight|.*bias')
      #
      # ====Parameters
      #
      # +select+:: (regexp) Regular expressions to match Parameters.
      #
      # ====Returns
      #
      # The filtered ParameterDict.
      #
      def collect_params(select = nil)
        ret = ParameterDict.new(prefix: @params.prefix)
        if select
          ret.update(@params.select { |k, v| k =~ select })
        else
          ret.update(@params)
        end
        @children.each_value do |child|
          ret.update(child.collect_params(select))
        end
        ret
      end

      # Registers block as a child of self. `Block` s assigned to self as
      # attributes will be registered automatically.
      def register_child(block, name=nil)
        name ||= @children.length
        @children[name] = block
      end

      # Override to implement forward computation using NDArray. Only
      # accepts positional arguments.
      #
      # ====Parameters
      #
      # +args+:: (array of NDArray) Input tensors.
      #
      def forward(*args)
        raise NotImplementedError
      end

      # Calls #forward. Only accepts positional arguments.
      alias call forward
    end

    # `HybridBlock` supports forwarding with both Symbol and NDArray.
    #
    # ...
    class HybridBlock < Block
      def initialize(*args, **kwargs)
        super
        @cached_graph = nil
      end

      def register_child(block, name=nil)
        unless block.is_a? HybridBlock
          raise ArgumentError,
                "Children of HybridBlock must also be HybridBlock, " +
                "but #{block} has type #{block.class}. If you are using " +
                "Sequential, please try HybridSequential instead."
        end
        super
        # _clear_cached_op
      end

      # Defines the forward computation. Arguments can be either Symbol
      # or NDArray.
      #
      # ====Parameters
      #
      # +args+:: (array of Symbol or NDArray) Input tensors.
      #
      def forward(*args)
        case args.first
        when MXNet::Symbol
          kwargs = {}
          hybrid_forward(MXNet::Symbol, *args, **kwargs)
        when MXNet::NDArray
          ctx = args.first.context
          begin
            kwargs = @reg_params.inject({}) do |acc, (i, j)|
              acc[i.to_sym] = j.data(ctx: ctx)
              acc
            end
          rescue DeferredInitializationError
            deferred_infer_shape(*args)
            @params.each do |_, param|
              # NOTE: invoking private method on Parameter
              param.send(:_finish_deferred_init)
            end
            retry
          end
          hybrid_forward(MXNet::NDArray, *args, **kwargs)
        else
          raise ArgumentError, 'only Symbol or NDArray are supported'
        end
      end

      # Override to construct symbolic graph for this Block.
      #
      # ====Parameters
      #
      # +args+:: (array of NDArray or Symbol) Input tensors.
      #
      #
      def hybrid_forward(clazz, *args)
        raise NotImplementedError
      end

      def infer_type(*args)
        infer_attrs('infer_type', 'dtype', *args)
      end

      def infer_shape(*args)
        infer_attrs('infer_shape', 'shape', *args)
      end

      def deferred_infer_shape(*args)
        infer_shape(*args)
      end

      private

      # Infer attributes.
      def infer_attrs(fn, attr, *args)
        inputs, output = get_graph(*args)
        args, _ = _flatten(args)
        arg_attrs, _, aux_attrs =
          output.send(fn, inputs.zip(args).inject({}) { |a, (i, j)| a[i.name] = j.send(attr) ; a })
        sdict = output.list_arguments.zip(arg_attrs).to_h
          .merge(output.list_auxiliary_states.zip(aux_attrs).to_h)
        collect_params.values.each do |value|
          value.send("#{attr}=", sdict[value.name.to_sym])
        end
      end

      def get_graph(*args)
        @cached_graph ||=
          begin
            args, @_in_format = _flatten(args)
            if args.length > 1
              inputs = (0 ... args.length).map {|i| MXNet::Symbol.var("data#{i}") }
            else
              inputs = [Symbol.var('data')]
            end
            # TODO: implement MXNet::Symbol::Group
            # grouped_inputs = _regroup(inputs, @_in_format)[0]

            params = @reg_params.inject({}) do |acc, (i, j)|
              acc[i.to_sym] = j.var
              acc
            end

            out = with_name_scope do
              # TODO: implement MXNet::Symbol::Group
              # hybrid_forward(MXNet::Symbol, *grouped_inputs, **params)
              hybrid_forward(MXNet::Symbol, *inputs, **params)
            end
            out, @_out_format = _flatten(out)

            # TODO: implement MXNet::Symbol::Group
            # [inputs, MXNet::Symbol::Group.new(out)]
            [inputs, out]
          end
      end

      private def _flatten(args)
        case args
        when MXNet::NDArray
          return [args], 0
        when MXNet::Symbol
          length = args.list_outputs.length
          length = 0 if length <= 1
          return [args], length
        when Array
          flat, fmts = [], []
          args.each do |i|
            arg, fmt = _flatten(args)
            flat.concat(arg)
            fmts << fmt
          end
          return flat, fmts
        end

        raise ArgumentError,
              "HybridBlock input must be (nested) array of Symbol or " +
              "NDArray, but got #{args} of type #{args.class}"
      end

      private def _regroup(args, fmt)
        case fmt
        when Integer
          return args[0], args[1..-1] if fmt == 0
          return args[0...fmt], args[fmt..-1]
        when Array
          ret = []
          fmt.each do |i|
            res, args = _regroup(args, i)
            ret << res
          end
          return ret, args
        end

        raise ArgumentError,
              "HybridBlock output must be (nested) array of Symbol or " +
              "NDArray, but got #{args} of type #{args.class}"
      end

      def clear_cacheed_op
        @cached_graph = nil
      end
    end
  end
end
