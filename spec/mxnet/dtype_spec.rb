require 'spec_helper'

RSpec.describe MXNet::DType do
  describe '.id2name' do
    def id2name(id)
      MXNet::DType.id2name(id)
    end

    specify do
      expect(id2name(0)).to eq(:float32)
    end
  end

  describe '.name2id' do
    def name2id(name)
      MXNet::DType.name2id(name)
    end

    specify do
      expect(name2id(:float32)).to eq(0)
    end

    specify do
      expect(name2id('float32')).to eq(0)
    end
  end

  describe '.available?' do
    def available?(dtype)
      MXNet::DType.available?(dtype)
    end

    specify do
      expect(available?(0)).to eq(true)
      expect(available?(:float32)).to eq(true)
      expect(available?("float32")).to eq(true)
      expect(available?(:invalid_dtype)).to eq(false)
    end

    specify do
      expect { available?(0.0) }.to raise_error(TypeError)
    end
  end
end

RSpec.describe 'dtype' do
  specify 'dtype is always symbol' do
    x = MXNet::NDArray.zeros([3])
    expect(x.dtype).to eq(:float32)
  end

  specify 'dtype can be specified as a string' do
    x = MXNet::NDArray.zeros([3], dtype: 'float32')
    expect(x.dtype).to eq(:float32)

    x = MXNet::NDArray.zeros([3], dtype: 'float64')
    expect(x.dtype).to eq(:float64)
  end

  specify 'dtype can be specified as an integer' do
    x = MXNet::NDArray.zeros([3], dtype: 'float32')
    expect(x.dtype).to eq(:float32)

    x = MXNet::NDArray.zeros([3], dtype: 'float64')
    expect(x.dtype).to eq(:float64)
  end
end
