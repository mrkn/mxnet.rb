require 'mxnet/gluon/block'
require 'mxnet/gluon/nn'

module MXNet::Gluon::NN
  ##
  # Applies an activation function.
  #
  class Activation < MXNet::Gluon::HybridBlock
    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +activation+:: (string)
    #                Name of activation function to use.
    #
    def initialize(activation, **kwargs)
      super(**kwargs)
      @activation = activation
    end

    def hybrid_forward(clazz, data)
      clazz.Activation(data, act_type: @activation)
    end

    private

    def hint
      @activation
    end
  end
end
