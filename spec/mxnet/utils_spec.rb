require 'spec_helper'

module MXNet
  ::RSpec.describe Utils do
    describe '.decompose_slice' do
      def decompose_slice(slice_like)
        MXNet::Utils.decompose_slice(slice_like)
      end

      specify do
        expect(decompose_slice(0..1)).to eq([0, 2, nil])
        expect(decompose_slice(0...1)).to eq([0, 1, nil])
        expect(decompose_slice(0..0)).to eq([0, 1, nil])
        expect(decompose_slice(0...0)).to eq([0, 0, nil])
        expect(decompose_slice(0..-1)).to eq([0, nil, nil])
        expect(decompose_slice(0...-1)).to eq([0, -1, nil])
        expect(decompose_slice(0..-2)).to eq([0, -1, nil])
        expect(decompose_slice(0...-2)).to eq([0, -2, nil])
        expect(decompose_slice((0...10).step(1))).to eq([0, 10, 1])
        expect(decompose_slice((0...10).step(2))).to eq([0, 10, 2])
        expect(decompose_slice((10..0).step(-1))).to eq([10, nil, -1])
        expect(decompose_slice((10...0).step(-1))).to eq([10, 0, -1])
      end
    end
  end
end
