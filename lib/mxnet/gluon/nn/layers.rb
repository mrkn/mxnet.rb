require 'mxnet/gluon/block'
require 'mxnet/gluon/nn'

module MXNet::Gluon::NN
  class Dense < MXNet::Gluon::HybridBlock
    def initialize(units, use_bias: true, in_units: 0, **kwargs)
      super(**kwargs)
      @units = units
      @in_units = in_units
      self.weight = params.get('weight', shape: [units, in_units])
      self.bias = use_bias ? params.get('bias', shape: [units]) : nil
    end
    def hybrid_forward(clazz, data, weight = nil, bias = nil, **kwargs)
      weight ||= kwargs[:weight]
      bias ||= kwargs[:bias]
      clazz.FullyConnected(data, weight, bias, no_bias: bias.nil?, num_hidden: @units)
    end
  end
end
