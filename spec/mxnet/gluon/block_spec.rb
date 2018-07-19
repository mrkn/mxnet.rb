require 'spec_helper'
require 'mxnet/gluon/block'
require 'mxnet/gluon/parameter'
require 'mxnet/ndarray'

RSpec.describe MXNet::Gluon::Block do
  describe 'assignment' do
    let(:block) do
      MXNet::Gluon::Block.new
    end
    it 'automagically creates a block accessor' do
      a = block.a = MXNet::Gluon::Block.new
      expect(block.a).to equal(a)
    end
    it 'automagically creates a parameter accessor' do
      b = block.b = MXNet::Gluon::Parameter.new('b')
      expect(block.b).to equal(b)
    end
  end
  describe '#new' do
    let(:block) do
      MXNet::Gluon::Block.new
    end
    it 'assigns a unique prefix' do
      expect(block.prefix).to match(/^block[0-9]+_$/)
      expect(block.prefix).not_to eq(MXNet::Gluon::Block.new.prefix)
    end
    context 'with prefix' do
      let(:block) do
        MXNet::Gluon::Block.new(prefix: 'foo')
      end
      it 'uses the assigned prefix' do
        expect(block.prefix).to eq('foo')
      end
    end
    context 'with params' do
      let(:params) do
        MXNet::Gluon::ParameterDict.new.tap do |params|
          params.get('foo')
        end
      end
      let(:block) do
        MXNet::Gluon::Block.new(params: params)
      end
      it 'shares the assigned params' do
        expect(block.params.get('foo')).to equal(params.get('foo'))
      end
    end
  end
  describe '#with_name_scope' do
    let(:block) do
      MXNet::Gluon::Block.new
    end
    it 'prepends prefixes to scoped blocks' do
      block.with_name_scope do
        block.foo = MXNet::Gluon::Block.new
        block.foo.with_name_scope do
          block.foo.bar = MXNet::Gluon::Block.new
        end
      end
      expect(block.foo.prefix).to match(/^block[0-9]+_block0_$/)
      expect(block.foo.bar.prefix).to match(/^block[0-9]+_block0_block0_$/)
    end
  end
  describe '#collect_params' do
    let(:block) do
      MXNet::Gluon::Block.new(prefix: 'block_').tap do |block|
        block.params.get('foo')
        block.params.get('bar')
        block.params.get('baz')
      end
    end
    let(:child) do
      MXNet::Gluon::Block.new(prefix: 'block_').tap do |block|
        block.params.get('qoz')
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
    it 'returns matching parameters from children' do
      block.qoz = child
      params = MXNet::Gluon::ParameterDict.new(prefix: 'block_').tap do |params|
        params.get('qoz')
      end
      expect(block.collect_params(/_q/)).to eq(params)
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
