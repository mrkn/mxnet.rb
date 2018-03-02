require "spec_helper"

RSpec.describe MXNet do
  it "has a version number" do
    expect(MXNet::VERSION).not_to be nil
  end
end
