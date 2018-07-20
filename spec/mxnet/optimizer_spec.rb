require 'spec_helper'
require 'mxnet/optimizer'

RSpec.describe MXNet::Optimizer do
  describe '.create' do
    it 'accepts a class' do
      opt = MXNet::Optimizer.create(MXNet::Optimizer::SGD)
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
    it 'accepts an instance' do
      opt = MXNet::Optimizer.create(MXNet::Optimizer::SGD.new)
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
    it 'accepts a string' do
      opt = MXNet::Optimizer.create('sgd')
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
    it 'accepts a symbol' do
      opt = MXNet::Optimizer.create(:sgd)
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
  end
end

RSpec.describe MXNet::Optimizer::SGD do
  describe '#update' do
    let(:optimizer) do
      MXNet::Optimizer::SGD.new(learning_rate: 0.1)
    end
    it 'updates the weight' do
      weight = MXNet::NDArray.array([1])
      gradient = MXNet::NDArray.array([0.5])
      expect(optimizer.update(0, weight, gradient, nil).as_scalar).to be_within(0.01).of(0.95)
    end
  end
end
