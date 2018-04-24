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

RSpec.describe MXNet::LRScheduler::MultiFactorScheduler do
  specify do
    expect {
      MXNet::LRScheduler::MultiFactorScheduler.new(step: 10, factor: 0.2)
    }.to raise_error(ArgumentError, /step must be an Array with more than 1 items/)
    expect {
      MXNet::LRScheduler::MultiFactorScheduler.new(step: [10], factor: 0.2)
    }.to raise_error(ArgumentError, /step must be an Array with more than 1 items/)
  end

  specify do
    expect {
      MXNet::LRScheduler::MultiFactorScheduler.new(step: [10, 5], factor: 0.2)
    }.to raise_error(ArgumentError, /step must be an increasing integer array/)
  end

  specify do
    expect {
      MXNet::LRScheduler::MultiFactorScheduler.new(step: [0, 1, 2], factor: 0.2)
    }.to raise_error(ArgumentError, /step must be greater than or equal to 1/)
  end

  specify do
    lrsched = MXNet::LRScheduler::MultiFactorScheduler.new(step: [30, 60, 90], factor: 0.2)
    lrsched.base_lr = 0.1

    expect(lrsched.step).to eq([30, 60, 90])
    expect(lrsched.factor).to eq(0.2)
    expect(lrsched.count).to eq(0)
    expect(lrsched.base_lr).to eq(0.1)

    expect(lrsched.update(1)).to eq(0.1)
    expect(lrsched.base_lr).to eq(0.1)
    expect(lrsched.count).to eq(0)

    expect(lrsched.update(30)).to eq(0.1)
    expect(lrsched.base_lr).to eq(0.1)
    expect(lrsched.count).to eq(0)

    expect(lrsched.update(31)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(30)

    expect(lrsched.update(32)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(30)

    expect(lrsched.update(60)).to be_within(1e-12).of(0.02)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.02)
    expect(lrsched.count).to eq(30)

    expect(lrsched.update(61)).to be_within(1e-12).of(0.004)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.004)
    expect(lrsched.count).to eq(60)

    expect(lrsched.update(62)).to be_within(1e-12).of(0.004)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.004)
    expect(lrsched.count).to eq(60)

    expect(lrsched.update(90)).to be_within(1e-12).of(0.004)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.004)
    expect(lrsched.count).to eq(60)

    expect(lrsched.update(91)).to be_within(1e-12).of(0.0008)
    expect(lrsched.base_lr).to be_within(1e-12).of(0.0008)
    expect(lrsched.count).to eq(90)
  end
end
