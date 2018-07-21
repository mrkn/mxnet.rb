require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::NN::Activation do
  describe '.new' do
    it 'accepts a string' do
      act = MXNet::Gluon::NN::Activation.new('relu')
      expect(act.forward(MXNet::NDArray.array([-1, 0, 1])))
        .to eq(MXNet::NDArray.array([0, 0, 1]))
    end

    it 'accepts a symbol' do
      act = MXNet::Gluon::NN::Activation.new(:relu)
      expect(act.forward(MXNet::NDArray.array([-1, 0, 1])))
        .to eq(MXNet::NDArray.array([0, 0, 1]))
    end
  end
end
