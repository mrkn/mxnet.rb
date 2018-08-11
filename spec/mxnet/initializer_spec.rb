require 'spec_helper'

RSpec.describe MXNet::Initializer do
  describe '.create' do
    it 'accepts a class' do
      init = described_class.create(MXNet::Initializer::Zero)
      expect(init).to be_a(MXNet::Initializer::Zero)
    end
    it 'accepts an instance' do
      init = described_class.create(MXNet::Initializer::Zero.new)
      expect(init).to be_a(MXNet::Initializer::Zero)
    end
    it 'accepts a string' do
      init = described_class.create('zeros')
      expect(init).to be_a(MXNet::Initializer::Zero)
    end
    it 'accepts a symbol' do
      init = described_class.create(:zeros)
      expect(init).to be_a(MXNet::Initializer::Zero)
    end
  end

  describe '#[]' do
    let(:initializer) do
      described_class.new
    end
    it 'calls init_array' do
      array = MXNet::NDArray.array(1)
      expect(initializer).to receive(:init_array).with(array)
      initializer[array]
    end
  end
end
