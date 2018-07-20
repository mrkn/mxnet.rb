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
        block[:foo] = MXNet::Gluon::Block.new
        block[:foo].with_name_scope do
          block[:foo][:bar] = MXNet::Gluon::Block.new
        end
      end
      expect(block[:foo].prefix).to match(/^block[0-9]+_block0_$/)
      expect(block[:foo][:bar].prefix).to match(/^block[0-9]+_block0_block0_$/)
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

    it 'returns all its parameters' do
      params = MXNet::Gluon::ParameterDict.new('block_')
      params.get('foo')
      params.get('bar')
      params.get('baz')
      expect(block.collect_params.keys).to eq(params.keys)
    end

    it 'returns the matching parameters' do
      params = MXNet::Gluon::ParameterDict.new("block_")
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

  context 'given a simple model' do
    before do
      stub_const 'Foo', Class.new(MXNet::Gluon::HybridBlock)

      Foo.class_eval do
        def initialize(**kwargs)
          super
          self[:c] = params.get('c', allow_deferred_init: true, dtype: nil)
        end

        def hybrid_forward(clazz, i, **kwargs)
          c = kwargs[:c]
          i + c
        end
      end
    end

    let(:foo) do
      Foo.new
    end

    describe '#infer_shape' do
      let(:data) do
        MXNet::NDArray.array([1, 2, 3, 4]).reshape([2, 2])
      end

      it 'should infer the shape' do
        foo.infer_shape(data)
        expect(foo[:c].shape).to eq(data.shape)
      end
    end

    describe '#infer_type' do
      let(:data) do
        MXNet::NDArray.array([1], dtype: :float16)
      end

      it 'should infer the type' do
        foo.infer_type(data)
        expect(foo[:c].dtype).to eq(data.dtype)
      end
    end
  end
end
