require 'spec_helper'
require 'mxnet/gluon/parameter'

RSpec.describe MXNet::Gluon::Parameter do
  describe '.new' do
    specify do
      expect(described_class.new('test', shape: 1).shape).to eq([1])
      expect(described_class.new('test', shape: [1]).shape).to eq([1])
    end
  end
  describe '#init' do
    context 'without deferred initialization' do
      let(:parameter) do
        described_class.new('foo', shape: [1]).tap do |parameter|
          parameter.init
        end
      end
      it 'initializes the data array' do
        expect(parameter.data).to be_a(MXNet::NDArray)
      end
      it 'initializes the grad array' do
        expect(parameter.grad).to be_a(MXNet::NDArray)
      end
      it 'attaches grads' do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          expect(p.grad).to eq(g)
        end
      end
    end
    context 'with deferred initialization' do
      let(:parameter) do
        described_class.new('foo', allow_deferred_init: true).tap do |parameter|
          parameter.init
          parameter.shape = [1]
          parameter.send(:finish_deferred_init)
        end
      end
      it 'initializes the data array' do
        expect(parameter.data).to be_a(MXNet::NDArray)
      end
      it 'initializes the grad array' do
        expect(parameter.grad).to be_a(MXNet::NDArray)
      end
      it 'attaches grads' do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          expect(p.grad).to eq(g)
        end
      end
    end
    context 'for "init"' do
      let(:parameter) do
        described_class.new('foo', shape: 1)
      end
      it 'accepts a class' do
        parameter.init(init: MXNet::Initializer::Zero)
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts an instance' do
        parameter.init(init: MXNet::Initializer::Zero.new)
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts a string' do
        parameter.init(init: 'zeros')
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts a symbol' do
        parameter.init(init: :zeros)
        expect(parameter.data.to_a).to eq([0])
      end
    end
    context 'for "default_init"' do
      let(:parameter) do
        described_class.new('foo', shape: 1)
      end
      it 'accepts a class' do
        parameter.init(default_init: MXNet::Initializer::Zero)
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts an instance' do
        parameter.init(default_init: MXNet::Initializer::Zero.new)
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts a string' do
        parameter.init(default_init: 'zeros')
        expect(parameter.data.to_a).to eq([0])
      end
      it 'accepts a symbol' do
        parameter.init(default_init: :zeros)
        expect(parameter.data.to_a).to eq([0])
      end
    end
  end
  describe '#list_ctx' do
    context 'without deferred initialization' do
      let(:parameter) do
        described_class.new('foo', shape: [1, 2]).tap do |parameter|
          parameter.init
        end
      end
      it 'returns the contexts' do
        expect(parameter.list_ctx).to eq([MXNet.cpu])
      end
    end
    context 'with deferred initialization' do
      let(:parameter) do
        described_class.new('foo', allow_deferred_init: true).tap do |parameter|
          parameter.init
        end
      end
      it 'returns the contexts' do
        expect(parameter.list_ctx).to eq([MXNet.cpu])
      end
    end
  end
  describe '#data' do
    let(:parameter) do
      described_class.new('foo', shape: [1])
    end
    it 'fails if the parameter has not been initialized' do
      expect{parameter.data}.to raise_error(RuntimeError)
    end
    it 'fails if the parameter has not been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.data(ctx: MXNet.gpu)}.to raise_error(RuntimeError)
    end
    it 'succeeds if the parameter has been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.data(ctx: MXNet.cpu)}.not_to raise_error
    end
    it 'returns the initialized data' do
      parameter.init
      expect(parameter.data).to be_a(MXNet::NDArray)
    end
  end
  describe '#grad' do
    let(:parameter) do
      described_class.new('foo', shape: [1])
    end
    it 'fails if the parameter has not been initialized' do
      expect{parameter.grad}.to raise_error(RuntimeError)
    end
    it 'fails if the parameter has not been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.grad(ctx: MXNet.gpu)}.to raise_error(RuntimeError)
    end
    it 'succeeds if the parameter has been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.grad(ctx: MXNet.cpu)}.not_to raise_error
    end
    it 'returns the initialized grad' do
      parameter.init
      expect(parameter.grad).to be_a(MXNet::NDArray)
    end
  end
  describe '#shape=' do
    context 'with no shape' do
      let(:parameter) do
        described_class.new('foo')
      end
      it 'assigns the shape' do
        parameter.shape = [1, 2]
        expect(parameter.shape).to eq([1, 2])
      end
    end
    context 'with incomplete shape' do
      let(:parameter) do
        described_class.new('foo', shape: [1, 0, 3])
      end
      it 'completes the shape' do
        parameter.shape = [1, 2, 3]
        expect(parameter.shape).to eq([1, 2, 3])
      end
    end
    context 'with shape' do
      let(:parameter) do
        described_class.new('foo', shape: [1, 2])
      end
      it 'raises an error' do
        expect{parameter.shape = [1, 3]}.to raise_error(RuntimeError)
      end
    end
  end
  describe '#==' do
    let(:parameter) do
      described_class.new('foo', shape: [1])
    end
    it 'is true if name and shape are equal' do
      expect(parameter == described_class.new('foo', shape: [1])).to be(true)
    end
    it 'is false if name is not equal' do
      expect(parameter == described_class.new('boo', shape: [1])).to be(false)
    end
    it 'is false if shape is not equal' do
      expect(parameter == described_class.new('foo', shape: [2])).to be(false)
    end
  end
end

RSpec.describe MXNet::Gluon::ParameterDict do
  describe '#get' do
    context 'without a shared dict' do
      let(:parameter_dict) do
        described_class.new
      end
      it 'creates a new parameter if not in dict' do
        expect(parameter_dict.get('foo')).to be_a(MXNet::Gluon::Parameter)
      end
      it 'retrieves a previously created parameter' do
        expect(parameter_dict.get('bar')).to equal(parameter_dict.get('bar'))
      end
      it 'uses keyword arguments to create a parameter' do
        expect(parameter_dict.get('baz', shape: [1, 1]).shape).to eq([1, 1])
      end
    end
    context 'with a shared dict' do
      let(:shared_dict) do
        described_class.new.tap do |shared_dict|
          shared_dict.get('foo')
        end
      end
      let(:parameter_dict) do
        described_class.new(shared: shared_dict).tap do |parameter_dict|
          parameter_dict.get('bar')
        end
      end
      it 'retrieves a parameter from the shared dict' do
        expect(parameter_dict.get('foo')).to equal(shared_dict.get('foo'))
      end
    end
  end
  describe '#update' do
    let(:parameter_dict) do
      described_class.new
    end
    let(:other_dict) do
      described_class.new.tap do |other_dict|
        other_dict.get('foo', shape: 1)
      end
    end
    it 'copies parameters into dict' do
      parameter_dict.update(other_dict)
      expect(parameter_dict.get('foo')).to equal(other_dict.get('foo'))
    end
    it 'fails if parameters already exist' do
      parameter_dict.get('foo')
      expect{parameter_dict.update(other_dict)}.to raise_error(ArgumentError)
    end
  end
  describe '#init' do
    let(:parameter_dict) do
      described_class.new(prefix: 'name').tap do |parameter_dict|
        parameter_dict.get('foo', shape: 1)
      end
    end
    it 'initializes all parameters' do
      parameter_dict.init
      expect(parameter_dict.get('foo').data).to be_a(MXNet::NDArray)
    end
  end
  describe '#==' do
    let(:parameter_dict) do
      described_class.new(prefix: 'name').tap do |parameter_dict|
        parameter_dict.get('test')
      end
    end
    it 'is true if prefix and items are equal' do
      other_dict = described_class.new(prefix: 'name').tap do |other_dict|
        other_dict.get('test')
      end
      expect(parameter_dict == other_dict).to be(true)
    end
    it 'is false if prefix is not equal' do
      other_dict = described_class.new(prefix: 'other').tap do |other_dict|
        other_dict.get('test')
      end
      expect(parameter_dict == other_dict).to be(false)
    end
    it 'is false items is not equal' do
      other_dict = described_class.new(prefix: 'name').tap do |other_dict|
        other_dict.get('other')
      end
      expect(parameter_dict == other_dict).to be(false)
    end
  end
end
