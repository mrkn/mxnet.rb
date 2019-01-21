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
    # Blocks can be nested recursively in a tree structure. You can
    # create and assign child blocks as regular attributes. Blocks
    # assigned this way will be registered and #collect_params will
    # collect their parameters recursively. You can also manually
    # register child blocks with #register_child.
    #
    #     class Model < MXNet::Gluon::Block
    #       def initialize(**kwargs)
    #         super(**kwargs)
    #         self.with_name_scope do
    #           self.dense0 = MXNet::Gluon::NN.Dense.new(20)
    #           self.dense1 = MXNet::Gluon::NN.Dense.new(20)
    #         end
    #       end
    #
    #       def forward(input)
    #         act = MXNet::Gluon::NN.Activation.new(:relu)
    #         input = act.(self.dense0.(input))
    #         act.(self.dense1.(input))
    #       end
    #     end
    #
    #     model = Model.new
    #     model.collect_params.init
    #     model.(...)
    #
    class Block
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +prefix+:: (string, optional)
      #            Prefix acts like a name space. All child blocks
      #            created inside the parent block's "name scope" (via
      #            #with_name_scope) will have the parent block's prefix
      #            in their name.
      # +params+:: (ParameterDict, optional)
      #            ParameterDict for sharing weights with the new
      #            Block.
      #
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

      ##
      # Initializes parameters of this block and its children.
      # Equivalent to `self.collect_params.init(...)`.
      #
      # ====Parameters
      #
      # +init+::         (Initializer, default +nil+)
      #                  The initializer to use.
      # +ctx+::          (Context or array of Contexts, default +nil+)
      #                  Desired contexts.
      # +verbose+::      (boolean, default false)
      #                  Whether to verbosely print out details on
      #                  initialization.
      # +force_reinit+:: (boolean, default false)
      #                  Whether to force re-initialization if parameter
      #                  is already initialized.
      #
      def init(init: nil, ctx: nil, verbose: false, force_reinit: false)
        collect_params.init(init: init, ctx: ctx, verbose: verbose, force_reinit: force_reinit)
      end

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

      protected def _collect_params_with_prefix(prefix = '')
        prefix += '.' unless prefix.empty?
        {}.tap do |hash|
          @reg_params.each do |key, param|
            hash["#{prefix}#{key}"] = param
          end
          @children.each do |name, child|
            hash.merge!(child._collect_params_with_prefix("#{prefix}#{name}"))
          end
        end
      end

      ##
      # Saves parameters to file.
      #
      # Note that this method only saves parameters, not model
      # structure. If you want to save model structures, use
      # HybridBlock#export.
      #
      # For reference see: "Saving and Loading Gluon Models"
      # (https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).
      #
      # ====Parameters
      #
      # +filename+:: (string)
      #              Path to file.
      #
      def save_parameters(filename)
        # NOTE: invoking private method on Parameter
        params = _collect_params_with_prefix
        params.transform_values! { |p| p.send(:_reduce) }
        MXNet::NDArray.save(filename, params)
      end

      ##
      # Loads parameters from file.
      #
      # For reference see: "Saving and Loading Gluon Models"
      # (https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).
      #
      # ====Parameters
      #
      # +filename+::      (string)
      #                   Path to file.
      # +ctx+::           (Context or array of Context, default cpu)
      #                   Context(s) to initialize loaded parameters on.
      # +allow_missing+:: (boolean, default +false+)
      #                   Whether to silently skip parameters not
      #                   present in the file.
      # +ignore_extra+::  (boolean, default +false+)
      #                   Whether to silently ignore parameters not
      #                   present in this block.
      #
      def load_parameters(filename, ctx = MXNet.cpu,
                          allow_missing: false,
                          ignore_extra: false)
        params = _collect_params_with_prefix
        loaded = MXNet::NDArray.load(filename)
        unless allow_missing
          params.each do |key, value|
            unless loaded.has_key?(key)
              raise RuntimeError,
                    "Parameter '#{key}' is missing in file '#{filename}'. " \
                    "Set allow_missing: true to ignore missing parameters."
            end
          end
        end
        unless ignore_extra
          loaded.each do |key, value|
            unless params.has_key?(key)
              raise RuntimeError,
                    "Parameter '#{key}' loaded from file '#{filename}' is " \
                    "not present in this block. Set ignore_extra: true to " \
                    "ignore extra parameters."
            end
          end
        end
        params.each do |key, value|
          # NOTE: invoking private method on Parameter
          params[key].send(:_load_init, loaded[key], ctx)
        end
      end

      # Registers block as a child of self. `Block` s assigned to self as
      # attributes will be registered automatically.
      def register_child(block, name=nil)
        name ||= @children.length.to_s
        @children[name] = block
      end

      # Activates or deactivates HybridBlock children recursively. Has
      # no effect on non-hybrid children.
      #
      # ====Parameters
      #
      # +active+:: (boolean, default +true+)
      #            Whether to turn hybridization on or off.
      #
      def hybridize(active: true, **kwargs)
        @children.each_value do |child|
          child.hybridize(active: active, **kwargs)
        end
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
      def call(*args)
        forward(*args)
      end
    end

    # `HybridBlock` supports forwarding with both Symbol and NDArray.
    #
    # ...
    class HybridBlock < Block
      def initialize(*args, **kwargs)
        super
        @cached_graph = nil
        @cached_op = nil
        @active = false
        @flags = {}
      end

      def __setattr__(name, value)
        super
        _clear_cached_op if value.is_a?(HybridBlock)
      end

      def register_child(block, name=nil)
        unless block.is_a? HybridBlock
          raise ArgumentError,
                "Children of HybridBlock must also be HybridBlock, " +
                "but #{block} has type #{block.class}. If you are using " +
                "Sequential, please try HybridSequential instead."
        end
        super
        _clear_cached_op
      end

      def hybridize(active: true, **kwargs)
        @active = active
        @flags = kwargs
        _clear_cached_op
        super
      end

      # Exports HybridBlock to JSON format that can be loaded by
      # SymbolBlock.import.
      #
      # ====Parameters
      #
      # +filename+:: (string)
      #              Path and base filename to which to save
      #              model. Two files, "[filename]-symbol.json" and
      #              "[filename]-NNNN.params" will be created, where
      #              +NNNN+ is the 4 digit epoch number.
      # +epoch+::    (integer, default +0+)
      #              Epoch number of saved model.
      #
      def export(filename, epoch: 0)
        unless @cached_graph
          raise RuntimeError,
                "Please call hybridize and then run forward " \
                "at least once before calling export."
        end
        output = @cached_graph[1]
        output.save('%s-symbol.json' % filename)
        arg_names = output.list_arguments
        aux_names = output.list_auxiliary_states
        args = {}
        collect_params.each do |name, param|
          # NOTE: invoking private method on Parameter
          if arg_names.include?(name.to_sym)
            args["arg:#{name}"] = param.send(:_reduce)
          elsif aux_names.include?(name.to_sym)
            args["aux:#{name}"] = param.send(:_reduce)
          end
        end
        MXNet::NDArray.save('%s-%04d.params' % [filename, epoch], args)
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
          kwargs = @reg_params.inject({}) do |acc, (i, j)|
            acc[i.to_sym] = j.var
            acc
          end
          self.with_name_scope do
            if kwargs.empty?
              hybrid_forward(MXNet::Symbol, *args)
            else
              hybrid_forward(MXNet::Symbol, *args, **kwargs)
            end
          end
        when MXNet::NDArray
          MXNet::Context.with(args.first.context) do |ctx|
            return _call_cached_op(*args) if @active
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
            if kwargs.empty?
              hybrid_forward(MXNet::NDArray, *args)
            else
              hybrid_forward(MXNet::NDArray, *args, **kwargs)
            end
          end
        else
          raise ArgumentError,
                'only Symbol or NDArray are supported, ' \
                "not #{args.first.class}"
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
        inputs, output = _get_graph(*args)
        args, _ = _flatten(args)
        arg_attrs, _, aux_attrs =
          output.send(fn, inputs.zip(args).inject({}) { |a, (i, j)| a[i.name] = j.send(attr) ; a })
        sdict = output.list_arguments.zip(arg_attrs).to_h
          .merge(output.list_auxiliary_states.zip(aux_attrs).to_h)
        collect_params.values.each do |value|
          value.send("#{attr}=", sdict[value.name.to_sym])
        end
      end

      private def _get_graph(*args)
        @cached_graph ||=
          begin
            args, @_in_format = _flatten(args)
            if args.length > 1
              inputs = (0 ... args.length).map {|i| MXNet::Symbol.var("data#{i}") }
            else
              inputs = [Symbol.var('data')]
            end
            grouped_inputs = _regroup(inputs, @_in_format)[0]

            params = @reg_params.inject({}) do |acc, (i, j)|
              acc[i.to_sym] = j.var
              acc
            end

            out = with_name_scope do
              hybrid_forward(MXNet::Symbol, *grouped_inputs, **params)
            end
            out, @_out_format = _flatten(out)

            [inputs, MXNet::Symbol.group(out)]
          end
      end

      private def _build_cache(*args)
        @cached_op ||=
          begin
            _, output = _get_graph(*args)
            inputs = collect_params.values
            [inputs, output, MXNet::CachedOp.new(output, **@flags)]
          end
      end

      protected def _call_cached_op(*args)
        inputs, _, cached_op = _build_cache(*args)
        begin
          data = inputs.map(&:data)
          cached_op.call(*(args + data))
        rescue MXNet::Gluon::DeferredInitializationError
          deferred_infer_shape(*args)
          inputs.each do |input|
            # NOTE: invoking private method on Parameter
            input.send(:_finish_deferred_init)
          end
          retry
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
            arg, fmt = _flatten(i)
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

      def _clear_cached_op
        @cached_graph = nil
        @cached_op = nil
      end
    end

    # A block constructed from a Symbol.
    #
    class SymbolBlock < MXNet::Gluon::HybridBlock
      # Imports model previously saved to JSON format by
      # HybridBlock#export as a SymbolBlock for use in Gluon.
      #
      # ====Parameters
      #
      # +filename+:: (string)
      #              Path and base filename from which to load
      #              model. Two files, "[filename]-symbol.json" and
      #              "[filename]-NNNN.params" will be loaded, where
      #              +NNNN+ is the 4 digit epoch number.
      # +inputs+::   (string or array of strings)
      #              Input names.
      # +epoch+::    (integer, default +0+)
      #              Epoch number of saved model.
      # +ctx+::      (Context or array of Context, default cpu)
      #              Context(s) to initialize loaded parameters on.
      #
      def self.import(filename, inputs, epoch: 0, ctx: MXNet.cpu,
                      allow_missing: false,
                      ignore_extra: false)
        output = MXNet::Symbol.load('%s-symbol.json' % filename)
        inputs = [inputs] unless inputs.is_a?(Array)
        inputs = inputs.map { |i| MXNet::Symbol.var(i) }
        SymbolBlock.new(output, inputs).tap do |block|
          if epoch
            filename = '%s-%04d.params' % [filename, epoch]
            arg_dict = MXNet::NDArray.load(filename)
            arg_dict = MXNet::Utils.transform_hash_keys(arg_dict) { |k| k.gsub(/^(arg:|aux:)/, '') }
            unless allow_missing
              block.params.keys.each do |key|
                unless arg_dict.has_key?(key.to_s)
                  raise RuntimeError,
                        "Parameter '#{key}' is missing in file '#{filename}'. " \
                        "Set allow_missing: true to ignore missing parameters."
                end
              end
            end
            unless ignore_extra
              arg_dict.keys.each do |key|
                unless block.params.keys.include?(key.to_s)
                  raise RuntimeError,
                        "Parameter '#{key}' loaded from file '#{filename}' is " \
                        "not present in this block. Set ignore_extra: true to " \
                        "ignore extra parameters."
                end
              end
            end
            arg_dict.each do |key, value|
              param = block.params.get(key)
              param.shape = value.shape
              # NOTE: invoking private method on Parameter
              param.send(:_load_init, value, ctx)
            end
          end
        end
      end

      # Creates a new instance.
      #
      # =====Parameters
      #
      # +output+:: (Symbol)
      #            The output.
      # +inputs+:: (array of Symbols)
      #            The output's arguments that should be used as inputs.
      # +params+:: (ParameterDict, default: +nil+)
      #            Dict of arguments and auxiliary states that are not
      #            inputs.
      #
      def initialize(output, inputs, params: nil)
        super(
          prefix: '',
          params: MXNet::Gluon::ParameterDict.new(prefix: '', shared: params)
        )
        input_names = inputs.map { |i| i.name.to_sym }
        output.list_arguments.each do |i|
          unless input_names.include?(i)
            self.params.get(i.to_s, allow_deferred_init: true)
          end
        end
        output.list_auxiliary_states.each do |i|
          unless input_names.include?(i)
            self.params.get(i.to_s, allow_deferred_init: true, grad_req: :null)
          end
        end
        @cached_graph = [inputs, output]
        len = lcp(@params.keys).length
        @reg_parameters =
          @params.inject({}) do |acc, (name, param)|
            acc[name[len..-1]] = param
            acc
          end
      end

      def forward(*args)
        case args.first
        when MXNet::Symbol
          @cached_graph[1].dup.tap do |output|
            kwargs = @cached_graph[0].zip(args).map { |k, v| [k.name, v] }.to_h
            # NOTE: invoking private method on Symbol
            output.send(:compose, **kwargs)
          end
        when MXNet::NDArray
          MXNet::Context.with(args.first.context) do |ctx|
            self._call_cached_op(*args)
          end
        else
          raise ArgumentError,
                "only Symbol or NDArray are supported, " \
                "not #{args.first.class}"
        end
      end

      def hybrid_forward(clazz, *args)
        raise NotImplementedError
      end

      private

      # Gets the longest common prefix of names.
      def lcp(names)
        case names.length
        when 0, 1
          names.first || ''
        else
          min, max = names.minmax
          i = min.length.times{ |i| break i if min[i] != max[i] }
          min[0...i]
        end
      end
    end
  end
end
