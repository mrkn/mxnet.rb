require 'spec_helper'

RSpec.describe MXNet::LRScheduler::FactorScheduler do
  specify do
    lrsched = MXNet::LRScheduler::FactorScheduler.new(step: 10, factor: 0.2)
    lrsched.base_lr = 0.1

    expect(lrsched.step).to eq(10)
    expect(lrsched.factor).to eq(0.2)
    expect(lrsched.count).to eq(0)
    expect(lrsched.base_lr).to eq(0.1)

    expect(lrsched.update(1)).to eq(0.1)
    expect(lrsched.base_lr).to eq(0.1)
    expect(lrsched.count).to eq(0)

    expect(lrsched.update(10)).to eq(0.1)
    expect(lrsched.base_lr).to eq(0.1)
    expect(lrsched.count).to eq(0)

    expect(lrsched.update(11)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(10)

    expect(lrsched.update(12)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(10)

    expect(lrsched.update(20)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(10)

    expect(lrsched.update(21)).to be_within(1e-12).of(0.004)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.004)
    expect(lrsched.count).to eq(20)
  end
end
