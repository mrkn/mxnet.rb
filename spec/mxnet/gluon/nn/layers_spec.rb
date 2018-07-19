require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::NN::Dense do
  describe '.new' do
    let(:layer) do
      MXNet::Gluon::NN::Dense.new(1, in_units: 2)
    end
    specify do
      expect(layer.class).to eq(MXNet::Gluon::NN::Dense)
    end
    it 'should set the weight and bias' do
      expect(layer[:weight].shape).to eq([1, 2])
      expect(layer[:bias].shape).to eq([1])
    end
  end
  describe '#collect_params' do
    let(:layer) do
      MXNet::Gluon::NN::Dense.new(1, in_units: 2, prefix: 'dense_')
    end
    it 'should return params for weight and bias' do
      params = MXNet::Gluon::ParameterDict.new(prefix: "dense_")
      params.get('weight', shape: [1, 2])
      params.get('bias', shape: [1])
      expect(layer.collect_params.keys).to eq(params.keys)
    end
  end
  describe '#forward' do
    context 'with input units specified' do
      let(:layer) do
        MXNet::Gluon::NN::Dense.new(1, in_units: 2).tap do |layer|
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
        MXNet::Gluon::NN::Dense.new(1).tap do |layer|
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
      MXNet::Gluon::NN::Dense.new(1)
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
