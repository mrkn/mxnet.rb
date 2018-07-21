require 'spec_helper'
require 'mxnet/gluon/parameter'
require 'mxnet/gluon/nn/layers'

RSpec.describe MXNet::Gluon::NN do
  describe '.Sequential' do
    let(:layer) do
      MXNet::Gluon::NN.Sequential
    end
    specify do
      expect(layer).to be_a(MXNet::Gluon::NN::Sequential)
    end
  end
  describe '.Dense' do
    let(:layer) do
      MXNet::Gluon::NN.Dense(1)
    end
    specify do
      expect(layer).to be_a(MXNet::Gluon::NN::Dense)
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Sequential do
  describe '#add' do
    let(:layer) do
      described_class.new
    end
    let(:block) do
      MXNet::Gluon::Block.new
    end
    it 'should register block as a child' do
      layer.add(block)
      expect(layer.children).to eq([block])
    end
  end
  describe '#forward' do
    let(:layer) do
      described_class.new
    end
    let(:block) do
      MXNet::Gluon::Block.new
    end
    it 'should run a forward pass on its children' do
      layer.add(block)
      expect(block).to receive(:forward)
      layer.forward(MXNet::Symbol.var('input'))
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Dense do
  describe '.new' do
    let(:layer) do
      described_class.new(1)
    end
    it 'should set the weight and bias' do
      expect(layer.weight.shape).to eq([1, 0])
      expect(layer.bias.shape).to eq([1])
    end
    context 'with `in_units: 2`' do
      let(:layer) do
        described_class.new(1, in_units: 2)
      end
      it 'should set the weight and bias' do
        expect(layer.weight.shape).to eq([1, 2])
        expect(layer.bias.shape).to eq([1])
      end
    end
    context 'with `use_bias: false`' do
      let(:layer) do
        described_class.new(1, use_bias: false)
      end
      it 'should disable bias' do
        expect(layer.bias).to be_nil
      end
    end
    context 'with `activation: :relu`' do
      let(:layer) do
        described_class.new(1, activation: :relu)
      end
      it 'add activation' do
        expect(layer.act).to be_a(MXNet::Gluon::NN::Activation)
      end
    end
  end
  describe '#collect_params' do
    let(:layer) do
      described_class.new(1, in_units: 2, prefix: 'dense_')
    end
    it 'should return params for weight and bias' do
      params = MXNet::Gluon::ParameterDict.new(prefix: 'dense_').tap do |params|
        params.get('weight', shape: [1, 2])
        params.get('bias', shape: [1])
      end
      expect(layer.collect_params).to eq(params)
    end
  end
  describe '#forward' do
    context 'with input units specified' do
      let(:layer) do
        described_class.new(1, in_units: 2).tap do |layer|
          layer.collect_params.init
        end
      end
      it 'runs a forward pass' do
        data = MXNet::NDArray.array([[2, 1]])
        expect(layer.forward(data)).to be_a(MXNet::NDArray)
      end
    end
    context 'with input units inferred' do
      let(:layer) do
        described_class.new(1).tap do |layer|
          layer.collect_params.init
        end
      end
      it 'runs a forward pass' do
        data = MXNet::NDArray.array([[2, 1]])
        expect(layer.forward(data)).to be_a(MXNet::NDArray)
      end
    end
  end
  describe '#hybrid_forward' do
    let(:layer) do
      described_class.new(1)
    end
    it 'accepts keyword arguments' do
      data = MXNet::NDArray.array([[2, 1]])
      kwargs = {
        weight: MXNet::NDArray.array([[0.5, 0.5]]),
        bias: MXNet::NDArray.array([-1])
      }
      expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a).to eq([[0.5]])
    end
    it 'accepts positional arguments' do
      data = MXNet::NDArray.array([[2, 1]])
      weight = MXNet::NDArray.array([[0.5, 0.5]])
      bias = MXNet::NDArray.array([-1])
      expect(layer.hybrid_forward(MXNet::NDArray, data, weight, bias).to_narray.to_a).to eq([[0.5]])
    end
  end
end
