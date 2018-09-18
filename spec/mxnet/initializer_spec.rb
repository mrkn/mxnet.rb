require 'spec_helper'
require 'mxnet/initializer'

RSpec.describe MXNet::Init::Xavier do
  specify do
    initializer = MXNet::Init::Xavier.new
    expect(initializer.rnd_type).to eq(:uniform)
    expect(initializer.factor_type).to eq(:avg)
    expect(initializer.magnitude).to eq(3)
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
