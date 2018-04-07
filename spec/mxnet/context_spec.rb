require 'spec_helper'

module MXNet
  ::RSpec.describe Context do
    describe '#==' do
      specify do
        x = Context.new(:cpu)
        expect(x).to eq(Context.new(:cpu))
      end
    end
  end
end
