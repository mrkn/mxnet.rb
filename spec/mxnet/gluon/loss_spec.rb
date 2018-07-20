require 'spec_helper'
require 'mxnet/gluon/loss'

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
