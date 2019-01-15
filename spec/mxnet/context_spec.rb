require 'spec_helper'

RSpec.describe MXNet::Context do
  describe '#==' do
    specify do
      x = MXNet::Context.new(:cpu)
      expect(x).to eq(MXNet::Context.new(:cpu))
    end
  end

  describe '.with' do
    specify do
      ctx = MXNet::gpu(0)
      expect {|b| MXNet::Context.with(ctx, &b) }.to yield_with_args(ctx)
    end

    specify do
      initial_context = MXNet::Context.default
      MXNet::Context.with(MXNet.gpu(0)) do
        expect(MXNet::Context.default).not_to eq(initial_context)
        expect(MXNet::Context.default).to eq(MXNet.gpu(0))
      end
    end
  end
end
