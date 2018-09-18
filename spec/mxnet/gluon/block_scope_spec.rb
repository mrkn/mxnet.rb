require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::BlockScope do
  describe '.make_prefix_and_params' do
    context 'when BlockScope.current is nil' do
      specify do
        prefix, params = MXNet::Gluon::BlockScope.make_prefix_and_params(nil, nil, 'foo')
        expect(prefix).to eq('foo0_')
        expect(params).to be_a(MXNet::Gluon::ParameterDict)
        expect(params.prefix).to eq('foo0_')

        prefix, params = MXNet::Gluon::BlockScope.make_prefix_and_params('bar_', nil, 'foo')
        expect(prefix).to eq('bar_')
        expect(params).to be_a(MXNet::Gluon::ParameterDict)
        expect(params.prefix).to eq('bar_')

        params_baz = MXNet::Gluon::ParameterDict.new('baz_')
        prefix, params = MXNet::Gluon::BlockScope.make_prefix_and_params('bar_', params_baz, 'foo')
        expect(prefix).to eq('bar_')
        expect(params.shared).to equal(params_baz)
      end

      context 'when entering prefixed name manager' do
        specify do
          MXNet::Name::Prefix.new(prefix: 'ahi_').enter do
            prefix, _ = MXNet::Gluon::BlockScope.make_prefix_and_params(nil, nil, 'foo')
            expect(prefix).to eq('ahi_foo0_')
          end
        end
      end
    end

    context 'when BlockScope.current is given' do
      let(:parent_params) do
        MXNet::Gluon::ParameterDict.new('parent_')
      end

      let(:block) do
        MXNet::Gluon::Block.new(params: parent_params)
      end

      let(:block_scope) do
        MXNet::Gluon::BlockScope.new(block)
      end

      before do
        block_scope.enter
      end

      after do
        block_scope.exit
      end

      specify do
        parent_params[:parent_entry] = 42
        prefix, params = MXNet::Gluon::BlockScope.make_prefix_and_params(nil, nil, 'foo')
        expect(prefix).to eq('block0_foo0_')
        expect(params.prefix).to eq('parent_foo0_')
        expect(params.shared).to equal(parent_params)

        prefix, params = MXNet::Gluon::BlockScope.make_prefix_and_params(nil, nil, 'foo')
        expect(prefix).to eq('block0_foo1_')
        expect(params.prefix).to eq('parent_foo1_')
        expect(params.shared).to equal(parent_params)
      end
    end
  end

  describe '#enter and #exit' do
    let(:block) { MXNet::Gluon::Block.new(prefix: 'prefix_') }
    subject(:block_scope) { MXNet::Gluon::BlockScope.new(block) }

    context 'block is empty prefix' do
      let(:block) { MXNet::Gluon::Block.new(prefix: '') }

      specify do
        expect(MXNet::Gluon::BlockScope.current).to equal(nil)
        block_scope.enter
        expect(MXNet::Gluon::BlockScope.current).to equal(nil)
      end
    end

    specify do
      expect(MXNet::Gluon::BlockScope.current).to equal(nil)

      block_scope.enter
      expect(MXNet::Gluon::BlockScope.current).to equal(block_scope)

      block_scope2 = MXNet::Gluon::BlockScope.new(block)
      block_scope2.enter
      expect(MXNet::Gluon::BlockScope.current).to equal(block_scope2)

      block_scope2.exit
      expect(MXNet::Gluon::BlockScope.current).to equal(block_scope)

      block_scope.exit
      expect(MXNet::Gluon::BlockScope.current).to equal(nil)
    end
  end

  describe '#enter with block' do
    specify do
      block = MXNet::Gluon::Block.new(prefix: 'foo_')
      block_scope = MXNet::Gluon::BlockScope.new(block)
      expect { |b|
        block_scope.enter do
          b.to_proc.call
          expect(MXNet::Gluon::BlockScope.current).to equal(block_scope)
        end
        expect(MXNet::Gluon::BlockScope.current).to equal(nil)
      }.to yield_control
    end
  end
end
