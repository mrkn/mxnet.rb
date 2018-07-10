require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Block do
  describe '#collect_params' do
    let(:block) do
      MXNet::Gluon::Block.new.tap do |block|
        block.params.get('foo')
        block.params.get('bar')
        block.params.get('baz')
      end
    end

    it 'returns all its parameters' do
      params = MXNet::Gluon::ParameterDict.new(prefix: 'block_').tap do |params|
        params.get('foo')
        params.get('bar')
        params.get('baz')
      end
      expect(block.collect_params).to eq(params)
    end

    it 'returns the matching parameters' do
      params = MXNet::Gluon::ParameterDict.new(prefix: 'block_').tap do |params|
        params.get('bar')
        params.get('baz')
      end
      expect(block.collect_params(/_ba/)).to eq(params)
    end
  end

  describe '#forward' do
    let(:block) do
      MXNet::Gluon::Block.new
    end

    it 'is not implemented' do
      expect{block.forward(MXNet::NDArray.array([]))}.to raise_error(NotImplementedError)
    end
  end
end

RSpec.describe MXNet::Gluon::HybridBlock do
  describe '#forward' do
    let(:block) do
      MXNet::Gluon::HybridBlock.new
    end

    it 'is not implemented' do
      expect{block.forward(MXNet::NDArray.array([]))}.to raise_error(NotImplementedError)
    end
  end
end
