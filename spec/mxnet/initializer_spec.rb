require 'spec_helper'

RSpec.describe MXNet::Initializer do
  describe '#[]' do
    let(:initializer) do
      MXNet::Initializer.new
    end
    it 'calls init_array' do
      array = MXNet::NDArray.array(1)
      expect(initializer).to receive(:init_array).with(array)
      initializer[array]
    end
  end
end
