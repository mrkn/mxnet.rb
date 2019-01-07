require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::NN::Sequential do
  describe '#add' do
    let(:layer) do
      MXNet::Gluon::NN::Sequential.new
    end

    it 'should register block as a child' do
      layer.add(MXNet::Gluon::NN::Dense.new(1))
      expect(layer.collect_params.keys).to contain_exactly('dense0_weight', 'dense0_bias')
    end
  end

  describe '#forward' do
    let(:layer) do
      MXNet::Gluon::NN::Sequential.new
    end

    let(:block) do
      MXNet::Gluon::Block.new
    end

    it 'should run a forward pass on its children' do
      mock_result = Object.new
      symbol = MXNet::Symbol.var('input')
      expect(block).to receive(:forward).with(symbol).and_return(mock_result)
      layer.add(block)
      expect(layer.forward(symbol)).to be_equal(mock_result)
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Dense do
  describe '.new' do
    let(:layer) do
      MXNet::Gluon::NN::Dense.new(1)
    end

    it 'should set the weight and bias' do
      expect(layer[:weight].shape).to eq([1, 0])
      expect(layer[:bias].shape).to eq([1])
    end

    context 'for "in_units"' do
      let(:layer) do
        MXNet::Gluon::NN::Dense.new(1, in_units: 2)
      end

      it 'should set the weight and bias' do
        expect(layer[:weight].shape).to eq([1, 2])
        expect(layer[:bias].shape).to eq([1])
      end
    end

    context 'for "use_bias"' do
      let(:layer) do
        MXNet::Gluon::NN::Dense.new(1, use_bias: false)
      end

      it 'should disable bias' do
        expect(layer[:bias]).to be_nil
      end
    end

    context 'for "activation"' do
      let(:layer) do
        MXNet::Gluon::NN::Dense.new(1, activation: :relu)
      end

      it 'add activation' do
        expect(layer[:act]).to be_a(MXNet::Gluon::NN::Activation)
      end
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

    let(:data) do
      MXNet::NDArray.array([[2, 1]])
    end

    let(:weight) do
      MXNet::NDArray.array([[0.5, 0.5]])
    end

    let(:bias) do
      MXNet::NDArray.array([-1])
    end

    let(:kwargs) do
      {weight: weight, bias: bias}
    end

    let(:output) do
      [[0.5]]
    end

    it 'accepts keyword arguments' do
      expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a)
        .to eq(output)
    end

    it 'accepts positional arguments' do
      expect(layer.hybrid_forward(MXNet::NDArray, data, weight, bias).to_narray.to_a)
        .to eq(output)
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Internal::Conv do
  let(:newargs) do
    {channels: 1, kernel_size: [2, 2], strides: 1, padding: 0, dilation: 1, layout: 'NCHW'}
  end

  describe '.new' do
    let(:layer) do
      described_class.new(**newargs)
    end

    it 'should set the weight and bias' do
      expect(layer[:weight].shape).to eq([1, 0, 2, 2])
      expect(layer[:bias].shape).to eq([1])
    end

    context 'with `in_channels: 3`' do
      let(:layer) do
        described_class.new(**newargs.merge(in_channels: 3))
      end

      it 'should set the weight and bias' do
        expect(layer[:weight].shape).to eq([1, 3, 2, 2])
        expect(layer[:bias].shape).to eq([1])
      end
    end

    context 'with `use_bias: false`' do
      let(:layer) do
        described_class.new(**newargs.merge(use_bias: false))
      end

      it 'should disable bias' do
        expect(layer[:bias]).to be_nil
      end
    end

    context 'with `activation: :relu`' do
      let(:layer) do
        described_class.new(**newargs.merge(activation: :relu))
      end

      it 'add activation' do
        expect(layer[:act]).to be_a(MXNet::Gluon::NN::Activation)
      end
    end
  end

  describe '#hybrid_forward' do
    let(:layer) do
      described_class.new(**newargs)
    end

    let(:data) do
      MXNet::NDArray.arange(16).reshape([1, 1, 4, 4])
    end

    let(:weight) do
      MXNet::NDArray.ones(4).reshape([1, 1, 2, 2])
    end

    let(:bias) do
      MXNet::NDArray.zeros(1)
    end

    let(:kwargs) do
      {weight: weight, bias: bias}
    end

    let(:output) do
      [[[[10, 14, 18],
         [26, 30, 34],
         [42, 46, 50]
        ]]]
    end

    it 'accepts keyword arguments' do
      expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a)
        .to eq(output)
    end

    it 'accepts positional arguments' do
      expect(layer.hybrid_forward(MXNet::NDArray, data, weight, bias).to_narray.to_a)
        .to eq(output)
    end

    context 'with `strides: 2`' do
      let(:layer) do
        described_class.new(**newargs.merge(strides: 2))
      end

      let(:output) do
        [[[[10, 18],
           [42, 50]
          ]]]
      end

      it 'convolves with a stride of 2' do
        expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a)
          .to eq(output)
      end
    end

    context 'with `padding: 1`' do
      let(:layer) do
        described_class.new(**newargs.merge(padding: 1))
      end

      let(:output) do
        [[[[ 0,  1,  3,  5,  3],
           [ 4, 10, 14, 18, 10],
           [12, 26, 30, 34, 18],
           [20, 42, 46, 50, 26],
           [12, 25, 27, 29, 15]
          ]]]
      end

      it 'pads the input by 1' do
        expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a)
          .to eq(output)
      end
    end

    context 'with `dilation: 2`' do
      let(:layer) do
        described_class.new(**newargs.merge(dilation: 2))
      end

      let(:output) do
        [[[[20, 24],
           [36, 40]
          ]]]
      end

      it 'dilation at a rate of 2' do
        expect(layer.hybrid_forward(MXNet::NDArray, data, kwargs).to_narray.to_a)
          .to eq(output)
      end
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Internal::Pooling do
  let(:newargs) do
    {pool_size: [2, 2], strides: 2, padding: 0}
  end

  describe '#hybrid_forward' do
    let(:layer) do
      described_class.new(**newargs)
    end

    let(:data) do
      MXNet::NDArray.arange(16).reshape([1, 1, 4, 4])
    end

    let(:output) do
      [[[[ 5,  7],
         [13, 15]
        ]]]
    end

    it 'pools the input' do
      expect(layer.hybrid_forward(MXNet::NDArray, data).to_narray.to_a)
        .to eq(output)
    end

    context 'with `strides: 1`' do
      let(:layer) do
        described_class.new(**newargs.merge(strides: 1))
      end

      let(:output) do
        [[[[ 5,  6,  7],
           [ 9, 10, 11],
           [13, 14, 15]
          ]]]
      end

      it 'pools with stride of 1' do
        expect(layer.hybrid_forward(MXNet::NDArray, data).to_narray.to_a)
          .to eq(output)
      end
    end

    context 'with `padding: 1`' do
      let(:layer) do
        described_class.new(**newargs.merge(padding: 1))
      end

      let(:output) do
        [[[[ 0,  2,  3],
           [ 8, 10, 11],
           [12, 14, 15]
          ]]]
      end

      it 'pools with padding of 1' do
        expect(layer.hybrid_forward(MXNet::NDArray, data).to_narray.to_a)
          .to eq(output)
      end
    end
  end
end

RSpec.describe MXNet::Gluon::NN::Flatten do
  describe '#hybrid_forward' do
    let(:layer) do
      described_class.new
    end

    let(:data) do
      MXNet::NDArray.arange(16).reshape([1, 1, 4, 4])
    end

    let(:output) do
      [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]
    end

    it 'flattens the input' do
      expect(layer.hybrid_forward(MXNet::NDArray, data).to_narray.to_a)
        .to eq(output)
    end
  end
end
