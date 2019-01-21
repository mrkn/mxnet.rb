require 'spec_helper'
require 'mxnet/initializer'

RSpec.describe MXNet::Init do
  describe '.create' do
    it 'does not accept a class' do
      expect {
        MXNet::Init.create(MXNet::Init::Zero)
      }.to raise_error(ArgumentError)
    end
    it 'accepts an instance and returns it' do
      init0 = MXNet::Init::Zero.new
      init1 = MXNet::Init.create(init0)
      expect(init1).to be_equal(init0)
    end
    it 'accepts a string' do
      init = MXNet::Init.create('zeros')
      expect(init).to be_a(MXNet::Init::Zero)
    end
    it 'accepts a symbol' do
      init = MXNet::Init.create(:zeros)
      expect(init).to be_a(MXNet::Init::Zero)
    end
  end
end

RSpec.describe MXNet::Init::Xavier do
  specify do
    initializer = MXNet::Init::Xavier.new
    expect(initializer.rnd_type).to eq(:uniform)
    expect(initializer.factor_type).to eq(:avg)
    expect(initializer.magnitude).to eq(3)
  end

  specify 'registered' do
    expect {
      init = MXNet::Init.create(:xavier)
      expect(init).to be_instance_of(MXNet::Init::Xavier)
    }.not_to raise_error
  end

  specify 'weight' do
    initializer = MXNet::Init::Xavier.new(rnd_type: :gaussian, factor_type: :in, magnitude: 2)
    weight = MXNet::NDArray.zeros([4, 3])
    initializer.(MXNet::Init::InitDesc.new(:weight), weight)
    expect(weight.reshape([-1]).to_a).to include(be_nonzero)
  end

  specify 'bias' do
    initializer = MXNet::Init::Xavier.new(rnd_type: :gaussian, factor_type: :in, magnitude: 2)
    bias   = MXNet::NDArray.ones([4])
    initializer.(MXNet::Init::InitDesc.new(:bias), bias)
    expect(bias.to_a).to all(be_zero)
  end
end
