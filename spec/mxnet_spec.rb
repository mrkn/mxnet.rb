require "spec_helper"

RSpec.describe MXNet do
  it "has a version number" do
    expect(MXNet::VERSION).not_to be nil
  end

  describe '.var' do
    specify do
      x = MXNet.var(:x)
      expect(x).to be_a(MXNet::Symbol)
      expect(x.name).to eq(:x)
    end
  end

  describe MXNet::Variable do
    specify do
      expect(MXNet::Variable).to eq(MXNet::Symbol)
    end
  end
end
