require 'mxnet/gluon'

module MXNet
  module Gluon
    class BlockScope
      @current = nil

      class << self
        attr_accessor :current
      end

      # Create prefix and params for new `Block`.
      def self.make_prefix_and_params(prefix, params, hint)
        unless @current
          prefix ||= MXNet::Name::NameManager.current.get(nil, hint) + '_'
          if params
            params = ParameterDict.new(params.prefix, params)
          else
            params = ParameterDict.new(prefix)
          end
          return [prefix, params]
        end

        unless prefix
          count = @current.counter[hint] || 0
          prefix = "#{hint}#{count}_"
          @current.counter[hint] = count + 1
        end
        if params
          params = ParameterDict.new(params.prefix, params)
        else
          parent = @current.block.params
          params = ParameterDict.new(parent.prefix + prefix, parent.shared)
        end
        [@current.block.prefix + prefix, params]
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
        @children = []
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
          if existing.is_a?(Block)
            @children.each_with_index do |c, i|
              @children[i] = value if c.equal?(existing)
            end
          elsif value.is_a?(Block)
            register_child(value)
          end
        elsif value.is_a?(Block)
          register_child(value)
        end
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
      def name_scope
      end

      attr_reader :params
    end

    # `HybridBlock` supports forwarding with both Symbol and NDArray.
    #
    # ...
    class HybridBlock < Block
    end
  end
end
