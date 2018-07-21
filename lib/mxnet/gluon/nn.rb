require 'mxnet/gluon'

module MXNet
  module Gluon
    module NN
      class Sequential < Block
      end

      class HybridSequential < HybridBlock
      end
    end
  end
end

require_relative 'nn/activations'
require_relative 'nn/layers'
