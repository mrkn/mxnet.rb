require 'spec_helper'
require 'mxnet/gluon'

::RSpec.describe MXNet::Gluon::NN::Sequential do
  specify do
    nn = MXNet::Gluon::NN
    net = nn::Sequential.new
    net.with_name_scope do
      net << nn::Dense.new(128, activation: :relu)
      net << nn::Dense.new(64, activation: :relu)
      net << nn::Dense.new(10)
    end
  end
end
