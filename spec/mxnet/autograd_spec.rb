require 'spec_helper'

RSpec.describe MXNet::Autograd do
  describe '#record' do
    specify do
      expect { |b| MXNet::Autograd.record(&b) }.to yield_control
    end
  end
end
