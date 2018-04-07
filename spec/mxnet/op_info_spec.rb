require 'spec_helper'

module MXNet
  ::RSpec.describe OpInfo do
    describe '.lookup' do
      specify do
        expect {
          desc = OpInfo.lookup(MXNet::Symbol::Ops, :zeros_like)
          expect(desc.name).to eq(:zeros_like)
        }.not_to raise_error
      end

      specify do
        expect {
          OpInfo.lookup(MXNet::Symbol::Ops, :invalid_op_name)
        }.to raise_error(ArgumentError)
      end

      specify do
        expect {
          OpInfo.lookup(Module.new, :zeros_like)
        }.to raise_error(TypeError)
      end
    end
  end
end
