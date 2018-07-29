require 'spec_helper'
require 'tempfile'
require 'mxnet/gluon/block'
require 'mxnet/gluon/parameter'
require 'mxnet/ndarray'

RSpec.describe MXNet::Gluon::Block do
  describe 'assignment' do
    let(:block) do
      described_class.new
    end
    it 'raises exception if accessor is undefined' do
      expect{block.b}.to raise_error(NoMethodError)
    end
    it 'raises exception if called with too many arguments' do
      expect{block.send(:b, 1)}.to raise_error(ArgumentError)
      expect{block.send(:b=)}.to raise_error(ArgumentError)
    end
    it 'automagically defines a block accessor' do
      a = block.a = described_class.new
      expect(block.a).to equal(a)
    end
    it 'automagically defines a parameter accessor' do
      b = block.b = MXNet::Gluon::Parameter.new('b')
      expect(block.b).to equal(b)
    end
  end
  describe '#new' do
    let(:block) do
      described_class.new
    end
    it 'assigns a unique prefix' do
      expect(block.prefix).to match(/^block[0-9]+_$/)
      expect(block.prefix).not_to eq(described_class.new.prefix)
    end
    context 'with prefix' do
      let(:block) do
        described_class.new(prefix: 'foo')
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
        described_class.new(params: params)
      end
      it 'shares the assigned params' do
        expect(block.params.get('foo')).to equal(params.get('foo'))
      end
    end
  end
  describe '#with_name_scope' do
    let(:block) do
      described_class.new
    end
    it 'prepends prefixes to scoped blocks' do
      block.with_name_scope do
        block.foo = described_class.new
        block.foo.with_name_scope do
          block.foo.bar = described_class.new
        end
      end
      expect(block.foo.prefix).to match(/^block[0-9]+_block0_$/)
      expect(block.foo.bar.prefix).to match(/^block[0-9]+_block0_block0_$/)
    end
  end
  describe '#init' do
    let(:block) do
      described_class.new.tap do |block|
        block.params.get('foo', shape: 1)
      end
    end
    it 'initializes all parameters' do
      block.init
      expect(block.params.get('foo').data).to be_a(MXNet::NDArray)
    end
  end
  describe '#collect_params' do
    let(:block) do
      described_class.new(prefix: 'block_').tap do |block|
        block.params.get('foo')
        block.params.get('bar')
        block.params.get('baz')
      end
    end
    let(:child) do
      described_class.new(prefix: 'block_').tap do |block|
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
  describe '#save_parameters' do
    let(:file) do
      Tempfile.new('foo').path
    end
    let(:data) do
      ['120100000000000000000000000000000100000000000000c9fa93f900000000' \
       '0100000002000000000000000100000000000000000000000000000000000000' \
       '01000000000000000300000000000000666f6f'
      ].pack('H*').force_encoding('utf-8')
    end
    let(:block) do
      described_class.new.tap do |block|
        block.foo = block.params.get('foo', shape: [2], init: :zeros)
        block.init
      end
    end
    it 'creates a file with the parameter data' do
      block.save_parameters(file)
      expect(File.open(file).read).to eq(data)
    end
  end
  describe '#load_parameters' do
    let(:file) do
      Tempfile.new('foo').path
    end
    let(:data) do
      ['120100000000000000000000000000000100000000000000c9fa93f900000000' \
       '01000000020000000000000001000000000000000000000098f6543cdccbf63c' \
       '01000000000000000300000000000000666f6f'
      ].pack('H*').force_encoding('utf-8')
    end
    let(:block) do
      described_class.new.tap do |block|
        block.foo = block.params.get('foo', shape: [2], init: :zeros)
        block.init
      end
    end
    before do
      File.open(file, 'wb') { |f| f.write(data) }
    end
    it 'loads parameter data from a file' do
      block.load_parameters(file)
      expect(block.foo.data.to_a)
        .to match_array([
                          be_within(0.0001).of(0.0129982),
                          be_within(0.0001).of(0.0301265)
                        ])
    end
    context 'with mismatched parameters' do
      let(:block) do
        described_class.new.tap do |block|
          block.bar = block.params.get('bar', shape: [2], init: :zeros)
          block.init
        end
      end
      it 'to raise error about missing parameter' do
        expect{block.load_parameters(file, ignore_extra: true)}
          .to raise_error(RuntimeError, /allow_missing: true/)
      end
      it 'to raise error about extra parameter' do
        expect{block.load_parameters(file, allow_missing: true)}
          .to raise_error(RuntimeError, /ignore_extra: true/)
      end
    end
  end
  describe '#forward' do
    let(:block) do
      described_class.new
    end
    it 'is not implemented' do
      data = MXNet::NDArray.array([])
      expect{block.forward(data)}.to raise_error(NotImplementedError)
    end
  end
end

RSpec.describe MXNet::Gluon::HybridBlock do
  describe '#forward' do
    context 'when hybridized' do
      before do
        stub_const 'Foo', Class.new(described_class)
      end
      let(:foo) do
        Foo.new.tap do |foo|
          foo.hybridize
        end
      end
      it 'caches the forward operation' do
        expect(foo).to receive(:hybrid_forward).once.with(MXNet::Symbol, instance_of(MXNet::Symbol), {}) do |c, s|
          s * s
        end
        expect(foo.forward(MXNet::NDArray.array([1, 2, 3])).to_a).to eq([1, 4, 9])
        expect(foo.forward(MXNet::NDArray.array([2, 3, 1])).to_a).to eq([4, 9, 1])
        expect(foo.forward(MXNet::NDArray.array([3, 1, 2])).to_a).to eq([9, 1, 4])
      end
    end
  end
  context 'given a simple model' do
    before do
      stub_const 'Foo', Class.new(described_class)
      Foo.class_eval do
        def initialize(**kwargs)
          super
          self.c = params.get('c', allow_deferred_init: true, dtype: nil)
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
        expect(foo.c.shape).to eq(data.shape)
      end
    end
    describe '#infer_type' do
      let(:data) do
        MXNet::NDArray.array([1], dtype: :float16)
      end
      it 'should infer the type' do
        foo.infer_type(data)
        expect(foo.c.dtype).to eq(data.dtype)
      end
    end
  end
end
