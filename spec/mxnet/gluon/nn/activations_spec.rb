require 'spec_helper'
require 'mxnet/gluon/nn/activations'

RSpec.describe MXNet::Gluon::NN::Activation do
  describe '.new' do
    it 'accepts a string' do
      act = described_class.new('relu')
      expect(act.forward(MXNet::NDArray.array([-1, 0, 1])))
        .to eq(MXNet::NDArray.array([0, 0, 1]))
    end
    it 'accepts a symbol' do
      act = described_class.new(:relu)
      expect(act.forward(MXNet::NDArray.array([-1, 0, 1])))
        .to eq(MXNet::NDArray.array([0, 0, 1]))
    end
  end
end
