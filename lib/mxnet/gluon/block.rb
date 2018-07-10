require 'mxnet/gluon'
require 'mxnet/gluon/parameter'
require 'mxnet/ndarray'

module MXNet::Gluon
  ##
  # Scope for collecting child Blocks.
  #
  class BlockScope
    def self.create(prefix, params, hint)
      if prefix.nil?
        prefix = hint + '_'
      end
      if params.nil?
        params = ParameterDict.new(prefix: prefix)
      else
        params = ParameterDict.new(prefix: prefix, params: params)
      end
      [prefix, params]
    end
  end
  ##
  # Base class for all neural network layers and models. Your models
  # should subclass this class.
  #
  class Block
    def initialize(prefix: nil, params: nil, **kwargs)
      super()
      @prefix, @params = BlockScope.create(prefix, params, hint)
      @reg_parameters = {}
    end
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
      ret
    end
    ##
    # Calls #forward. Only accepts positional arguments.
    #
    #
    def [](*args)
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
    private
    def method_missing(name, value = nil)
      name = name.to_s
      if name[-1] == '='
        name = name[0...-1]
        case value
        when MXNet::Gluon::Block
        when MXNet::Gluon::Parameter
          @reg_parameters[name] = value
        else
          raise NoMethodError
        end
      else
        @reg_parameters[name] or
          raise NoMethodError
      end
    end
    def hint
      self.class.name.split('::').last.downcase
    end
  end
  ##
  # HybridBlock supports forwarding with both Symbol and NDArray.
  #
  class HybridBlock < Block
    def initialize(**kwargs)
      super(**kwargs)
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
        kwargs = {}
        hybrid_forward(MXNet::Symbol, *args, **kwargs)
      when MXNet::NDArray
        ctx = args.first.context
        kwargs = @reg_parameters.inject({}) do |acc, (i, j)|
          acc[i.to_sym] = j.data(ctx: ctx)
          acc
        end
        hybrid_forward(MXNet::NDArray, *args, **kwargs)
      else
        raise ArgumentError, 'only Symbol or NDArray are supported'
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
end
