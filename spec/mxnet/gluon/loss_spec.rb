require 'spec_helper'
require 'mxnet/gluon/loss'

RSpec.describe MXNet::Gluon::Loss::L1Loss do
  describe '#[]' do
    let(:loss) do
      described_class.new
    end

    it 'calculates L1 loss' do
      prediction = MXNet::NDArray.array([1])
      label = MXNet::NDArray.array([2])
      expect(loss.(prediction, label).as_scalar).to eq(1.0)
    end
  end
end

RSpec.describe MXNet::Gluon::Loss::L2Loss do
  describe '#hybrid_forward' do
    let(:loss) do
      MXNet::Gluon::Loss::L2Loss.new
    end
    it 'calculates L2 loss' do
      prediction = MXNet::NDArray.array([1])
      label = MXNet::NDArray.array([2])
      expect(loss.hybrid_forward(MXNet::NDArray, prediction, label).to_a).to eq([0.5])
    end
  end
end

RSpec.describe MXNet::Gluon::Loss::SoftmaxCrossEntropyLoss do
  describe '#[]' do
    let(:loss) do
      described_class.new
    end

    it 'calculates softmax cross-entropy loss' do
      prediction = MXNet::NDArray.array([0, 1, 0])
      label = MXNet::NDArray.array([1])
      expect(loss.(prediction, label).as_scalar).to be_within(0.001).of(0.551)
    end

    context 'when created with `sparse_label: false`' do
      let(:loss) do
        described_class.new(sparse_label: false)
      end

      it 'calculates softmax cross-entropy loss' do
        prediction = MXNet::NDArray.array([0, 1, 0])
        label = MXNet::NDArray.array([0, 1, 0])
        expect(loss.(prediction, label).as_scalar).to be_within(0.001).of(0.551)
      end
    end

    context 'when created with `from_logits: true`' do
      let(:loss) do
        described_class.new(from_logits: true)
      end

      it 'calculates softmax cross-entropy loss' do
        prediction = MXNet::NDArray.array([0, -1, 0])
        label = MXNet::NDArray.array([1])
        expect(loss.(prediction, label).as_scalar).to be_within(0.001).of(1.000)
      end
    end
  end
end
