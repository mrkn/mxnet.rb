require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Block do
  describe 'attri[ute assignment' do
    let(:block) do
      MXNet::Gluon::Block.new
    end

    it 'automagically creates a block accessor' do
      a = block[:a] = MXNet::Gluon::Block.new
      expect(block[:a]).to equal(a)
    end

    it 'automagically creates a parameter accessor' do
      b = block[:b] = MXNet::Gluon::Parameter.new('b')
      expect(block[:b]).to equal(b)
    end
  end

  describe '#collect_params' do
    let(:block) do
      MXNet::Gluon::Block.new.tap do |block|
        block.params.get('foo')
        block.params.get('bar')
        block.params.get('baz')
      end
    end

    it 'returns the matching parameters' do
      count = MXNet::Name::NameManager.current.next_count_for('block')
      params = MXNet::Gluon::ParameterDict.new("block#{count}_")
      params.get('bar')
      params.get('baz')
      expect(block.collect_params(/_ba/).keys).to eq(params.keys)
    end

    it 'returns matching parameters from children' do
      count = MXNet::Name::NameManager.current.next_count_for('block')
      child = MXNet::Gluon::Block.new
      child.params.get('qoz')

      block[:qoz] = child
      params = MXNet::Gluon::ParameterDict.new("block#{count}_")
      params.get('qoz')
      expect(block.collect_params(/_q/).keys).to eq(params.keys)
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
