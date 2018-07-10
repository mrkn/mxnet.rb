require 'spec_helper'
require 'mxnet/gluon/parameter'

RSpec.describe MXNet::Gluon::Parameter do
  describe '.new' do
    specify do
      expect(MXNet::Gluon::Parameter.new('test', shape: 1).shape).to eq([1])
      expect(MXNet::Gluon::Parameter.new('test', shape: [1]).shape).to eq([1])
    end
  end
  describe '#init' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('foo', shape: [1])
    end
    it 'initializes the data array' do
      parameter.init
      expect(parameter.data).to be_a(MXNet::NDArray)
    end
  end
  describe '#data' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('foo', shape: [1])
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
  describe '#==' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('foo', shape: [1])
    end
    it 'is true if name and shape are equal' do
      expect(parameter == MXNet::Gluon::Parameter.new('foo', shape: [1])).to be(true)
    end
    it 'is false if name is not equal' do
      expect(parameter == MXNet::Gluon::Parameter.new('boo', shape: [1])).to be(false)
    end
    it 'is false if shape is not equal' do
      expect(parameter == MXNet::Gluon::Parameter.new('foo', shape: [2])).to be(false)
    end
  end
end

RSpec.describe MXNet::Gluon::ParameterDict do
  describe '#get' do
    context 'without a shared dict' do
      let(:parameter_dict) do
        MXNet::Gluon::ParameterDict.new
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
        MXNet::Gluon::ParameterDict.new
      end
      let(:parameter_dict) do
        MXNet::Gluon::ParameterDict.new(shared: shared_dict).tap do |parameter_dict|
          shared_dict.get('foo')
        end
      end
      it 'retrieves a parameter from the shared dict' do
        expect(parameter_dict.get('foo')).to equal(shared_dict.get('foo'))
      end
    end
  end
  describe '#update' do
    let(:parameter_dict) do
      MXNet::Gluon::ParameterDict.new
    end
    let(:other_dict) do
      MXNet::Gluon::ParameterDict.new.tap do |other_dict|
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
      MXNet::Gluon::ParameterDict.new(prefix: 'name').tap do |parameter_dict|
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
      MXNet::Gluon::ParameterDict.new(prefix: 'name').tap do |parameter_dict|
        parameter_dict.get('test')
      end
    end
    it 'is true if prefix and items are equal' do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: 'name').tap do |other_dict|
        other_dict.get('test')
      end
      expect(parameter_dict == other_dict).to be(true)
    end
    it 'is false if prefix is not equal' do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: 'other').tap do |other_dict|
        other_dict.get('test')
      end
      expect(parameter_dict == other_dict).to be(false)
    end
    it 'is false items is not equal' do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: 'name').tap do |other_dict|
        other_dict.get('other')
      end
      expect(parameter_dict == other_dict).to be(false)
    end
  end
end
