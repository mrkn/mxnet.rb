require 'spec_helper'
require 'mxnet/optimizer'

RSpec.describe MXNet::SGD do
  describe '#update' do
    let(:optimizer) do
      MXNet::SGD.new(learning_rate: 0.1)
    end
    it 'updates the weight' do
      weight = MXNet::NDArray.array([1])
      gradient = MXNet::NDArray.array([0.5])
      expect(optimizer.update(0, weight, gradient, nil).as_scalar).to be_within(0.01).of(0.95)
    end
  end
end
