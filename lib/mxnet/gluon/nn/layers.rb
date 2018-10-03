require 'mxnet/gluon/block'
require 'mxnet/gluon/nn'

module MXNet::Gluon::NN
  class Dense < MXNet::Gluon::HybridBlock
    def initialize(units, use_bias: true, in_units: 0, **kwargs)
      super(**kwargs)
      with_name_scope do
        @units = units
        @in_units = in_units
        self[:weight] = params.get('weight', shape: [units, in_units])
        self[:bias] = use_bias ? params.get('bias', shape: [units]) : nil
      end
    end
    def hybrid_forward(clazz, data, weight=nil, bias = nil, **kwargs)
      if (weight ||= kwargs[:weight]).nil?
        raise ArgumentError, "weight is not given"
      end
      bias ||= kwargs[:bias]
      clazz.FullyConnected(data, weight, bias, no_bias: bias.nil?, num_hidden: @units)
    end
  end
end
