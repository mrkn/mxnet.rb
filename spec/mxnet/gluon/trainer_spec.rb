require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Trainer do
  describe '#update' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('p', shape: [1]).tap do |param|
        param.init
      end
    end
    let(:optimizer) do
      instance_double(MXNet::Optimizer::Base).tap do |opt|
        allow(opt).to receive(:is_a?).with(MXNet::Optimizer::Base).and_return(true)
        allow(opt).to receive(:rescale_grad=)
        allow(opt).to receive(:param_dict=)
      end
    end
    it 'calls Optimizer#update for every parameter' do
      p = parameter
      o = optimizer
      expect(o).to receive(:update_multi_precision).with(0, p.data, p.grad, nil).exactly(1).times
      trainer = MXNet::Gluon::Trainer.new({p: p}, o)
      trainer.update(1)
    end
  end
end
