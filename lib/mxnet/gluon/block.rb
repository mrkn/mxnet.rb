require 'mxnet/gluon'
require 'mxnet/gluon/parameter'
require 'mxnet/ndarray'

module MXNet
  module Gluon
    ##
    # Scope for collecting child Blocks.
    #
    class BlockScope
      def initialize(block = nil)
        @block = block
        @counters = Hash.new(-1)
      end

      attr_accessor :block
      attr_accessor :counters

      def self.create(prefix, params, hint)
        current =
          Thread.current['mxnet_gluon_blockscope_current'] ||=
          BlockScope.new
        if current.block
          if prefix.nil?
            prefix = "#{hint}#{current.counters[hint] += 1}_"
          end
          if params.nil?
            params = ParameterDict.new(prefix: "#{current.block.prefix}#{prefix}")
          else
            params = ParameterDict.new(prefix: "#{current.block.prefix}#{prefix}", shared: params)
          end
          ["#{current.block.prefix}#{prefix}", params]
        else
          if prefix.nil?
            prefix = "#{hint}#{current.counters[hint] += 1}_"
          end
          if params.nil?
            params = ParameterDict.new(prefix: prefix)
          else
            params = ParameterDict.new(prefix: params.prefix, shared: params)
          end
          [prefix, params]
        end
      end

      def call(block)
        previous = Thread.current['mxnet_gluon_blockscope_current']
        Thread.current['mxnet_gluon_blockscope_current'] = block.scope
        yield
      ensure
        Thread.current['mxnet_gluon_blockscope_current'] = previous
      end
    end

    ##
    # Base class for all neural network layers and models. Your models
    # should subclass this class.
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
    #           self.dense0 = MXNet::Gluon::NN.Dense(20)
    #           self.dense1 = MXNet::Gluon::NN.Dense(20)
    #         end
    #       end
    #       def forward(input)
    #         act = MXNet::Gluon::NN.Activation(:relu)
    #         input = act[self.dense0[input]]
    #         act[self.dense1[input]]
    #       end
    #     end
    #
    #     model = Model.new
    #     model.collect_params.init
    #     model[...]
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
        @scope = BlockScope.new(self)
        @prefix, @params = BlockScope.create(prefix, params, hint)
        @reg_parameters = {}
        @reg_children = {}
        @reg_other = {}
      end

      ##
      # Scope of this block.
      #
      attr_reader :scope

      ##
      # Prefix of this Block.
      #
      attr_reader :prefix

      ##
      # Returns this Block's ParameterDict (does not include its
      # children's parameters).
      #
      attr_reader :params

      ##
      # Enters a name space managing Block names.
      #
      #     self.with_name_scope do
      #       self.dense = MXNet::Gluon::NN.Dense(20)
      #     end
      #
      def with_name_scope(&proc)
        @scope.call(self, &proc)
      end

      ##
      # Returns this block's registered children.
      #
      def children
        @reg_children.values
      end

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
      # +force_reinit+:: (boolean, default false)
      #                  Whether to force re-initialization if parameter
      #                  is already initialized.
      #
      def init(init: nil, ctx: nil, force_reinit: false)
        collect_params.init(init: init, ctx: ctx, force_reinit: force_reinit)
      end

      ##
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
        @reg_children.each do |_, child|
          ret.update(child.collect_params(select))
        end
        ret
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
        params = collect_params_for_storage.transform_values { |p| p.send(:reduce) }
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
        params = collect_params_for_storage
        loaded = MXNet::NDArray.load(filename)
        unless allow_missing
          params.each do |key, value|
            unless loaded.key?(key)
              raise RuntimeError,
                    "Parameter '#{key}' is missing in file '#{filename}'. " \
                    "Set allow_missing: true to ignore missing parameters."
            end
          end
        end
        unless ignore_extra
          loaded.each do |key, value|
            unless params.key?(key)
              raise RuntimeError,
                    "Parameter '#{key}' loaded from file '#{filename}' is " \
                    "not present in this block. Set ignore_extra: true to " \
                    "ignore extra parameters."
            end
          end
        end
        params.each do |key, value|
          # NOTE: invoking private method on Parameter
          params[key].send(:load_and_init, ctx, loaded[key])
        end
      end

      ##
      # Registers block as a child of self. Blocks assigned as
      # attributes will be registered automatically.
      #
      def register_child(block, name = nil)
        name = @reg_children.length.to_s if name.nil?
        @reg_children[name] = block
      end

      ##
      # Activates or deactivates HybridBlock children recursively. Has
      # no effect on non-hybrid children.
      #
      # ====Parameters
      #
      # +active+:: (boolean, default +true+)
      #            Whether to turn hybridization on or off.
      #
      def hybridize(active: true, **kwargs)
        @reg_children.each do |_, child|
          child.hybridize(active: active, **kwargs)
        end
      end

      ##
      # Calls #forward. Only accepts positional arguments.
      #
      def call(*args)
        forward(*args)
      end

      ##
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

      protected

      def collect_params_for_storage(prefix = '')
        prefix += '.' unless prefix.empty?
        {}.tap do |hash|
          @reg_parameters.each do |key, param|
            hash[prefix + key] = param
          end
          @reg_children.each do |key, child|
            hash.merge!(child.collect_params_for_storage(prefix + key))
          end
        end
      end

      private

      def method_missing(sym, *args)
        name = sym.to_s
        if name[-1] == '='
          args.length == 1 or
            raise ArgumentError, "wrong number of arguments (#{args.length} for 1) to `#{sym}'"
          set_attr(name[0...-1], *args)
        else
          args.length == 0 or
            raise ArgumentError, "wrong number of arguments (#{args.length} for 0) to `#{sym}'"
          get_attr(name)
        end
      end

      def set_attr(name, value)
        case value
        when MXNet::Gluon::Block
          register_child(value, name)
        when MXNet::Gluon::Parameter
          @reg_parameters[name] = value
        else
          @reg_other[name] = value
        end
      end

      def get_attr(name)
        [@reg_children, @reg_parameters, @reg_other].each do |h|
          return h[name] if h.key?(name)
        end
        raise NoMethodError, "undefined method `#{name}' for #{self}"
      end

      def hint
        self.class.name.split('::').last.downcase
      end
    end

    ##
    # CachedGraph encapsulates caching symbolized operations.
    #
    module CachedGraph
      def initialize(**kwargs)
        super(**kwargs)
        clear_cache
      end

      def clear_cache
        @cached_graph = nil
        @cached_op = nil
        @flags = {}
      end

      def infer_type(*args)
        infer_attrs('infer_type', 'dtype', *args)
      end

      def infer_shape(*args)
        infer_attrs('infer_shape', 'shape', *args)
      end

      ##
      # Exports cached graph and parameters in a format that can be
      # loaded by SymbolBlock.import.
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
            args["arg:#{name}"] = param.send(:reduce)
          elsif aux_names.include?(name.to_sym)
            args["aux:#{name}"] = param.send(:reduce)
          end
        end
        MXNet::NDArray.save('%s-%04d.params' % [filename, epoch], args)
      end

      protected

      def call_cached(*args)
        inputs, _, cached_op = get_cached_op(*args)
        begin
          data = inputs.map(&:data)
          cached_op.call(*(args + data))
        rescue MXNet::Gluon::DeferredInitializationError
          infer_shape(*args)
          inputs.each do |input|
            # NOTE: invoking private method on Parameter
            input.send(:finish_deferred_init)
          end
          retry
        end
      end

      private

      ##
      # Infer attributes (type and shape).
      #
      def infer_attrs(fn, attr, *args)
        inputs, output = get_graph(*args)
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
            inputs =
              if args.length > 1
                (0...args.length).map do |i|
                  MXNet::Symbol.var("data#{i}")
                end
              else
                [MXNet::Symbol.var('data')]
              end
            params = @reg_parameters.inject({}) do |acc, (i, j)|
              acc[i.to_sym] = j.var
              acc
            end
            [inputs, hybrid_forward(MXNet::Symbol, *inputs, **params)]
          end
      end

      def get_cached_op(*args)
        @cached_op ||=
          begin
            _, output = get_graph(*args)
            inputs = collect_params.values
            [inputs, output, MXNet::CachedOp.new(output, @flags)]
          end
      end
    end

    ##
    # HybridBlock supports forwarding with both Symbol and NDArray.
    #
    class HybridBlock < Block
      include CachedGraph

      def initialize(**kwargs)
        super(**kwargs)
        @active = false
      end

      def register_child(block, name = nil)
        unless block.is_a?(MXNet::Gluon::HybridBlock)
          raise RuntimeError,
                "Children of a HybridBlock must also be a HybridBlock, " \
                "but #{block} has type #{block.class}. If you are using " \
                "Sequential, please try HybridSequential instead."
        end
        clear_cache
        super
      end

      def set_attr(name, value)
        clear_cache if value.is_a?(HybridBlock)
        super
      end

      def hybridize(active: true, **kwargs)
        clear_cache
        @active = active
        @flags = kwargs
        super
      end

      ##
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
          kwargs = @reg_parameters.inject({}) do |acc, (i, j)|
            acc[i.to_sym] = j.var
            acc
          end
          hybrid_forward(MXNet::Symbol, *args, **kwargs)
        when MXNet::NDArray
          if @active
            return call_cached(*args)
          end
          MXNet::Context.with(ctx = args.first.context) do
            begin
              kwargs = @reg_parameters.inject({}) do |acc, (i, j)|
                acc[i.to_sym] = j.data(ctx: ctx)
                acc
              end
            rescue MXNet::Gluon::DeferredInitializationError
              infer_shape(*args)
              @params.each do |_, param|
                # NOTE: invoking private method on Parameter
                param.send(:finish_deferred_init)
              end
              retry
            end
            hybrid_forward(MXNet::NDArray, *args, **kwargs)
          end
        else
          raise ArgumentError,
                "only Symbol or NDArray are supported, " \
                "not #{args.first.class}"
        end
      end

      ##
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
    end

    ##
    # A block constructed from a Symbol.
    #
    class SymbolBlock < MXNet::Gluon::Block
      include CachedGraph

      ##
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
            arg_dict = arg_dict.transform_keys { |k| k.gsub(/^(arg:|aux:)/, '') }
            unless allow_missing
              block.params.keys.each do |key|
                unless arg_dict.key?(key.to_s)
                  raise RuntimeError,
                        "Parameter '#{key}' is missing in file '#{filename}'. " \
                        "Set allow_missing: true to ignore missing parameters."
                end
              end
            end
            unless ignore_extra
              arg_dict.keys.each do |key|
                unless block.params.key?(key.to_sym)
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
              param.send(:load_and_init, ctx, value)
            end
          end
        end
      end

      ##
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
          MXNet::Context.with(args.first.context) do
            self.call_cached(*args)
          end
        else
          raise ArgumentError,
                "only Symbol or NDArray are supported, " \
                "not #{args.first.class}"
        end
      end

      private

      ##
      # Gets the longest common prefix of names.
      #
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

    def self.SymbolBlock(*args)
      SymbolBlock.new(*args)
    end
  end
end
