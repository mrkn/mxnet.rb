require 'spec_helper'

module MXNet
  ::RSpec.describe NDArray do
    describe '#to_narray' do
      specify do
        x = MXNet::NDArray.ones([2, 3])
        y = x.to_narray
        expect(Numo::NArray.array_type(y)).to eq(Numo::SFloat)
        expect(y).to eq(Numo::SFloat.ones(2, 3))

        x = MXNet::NDArray.ones([2, 3], dtype: :float64)
        y = x.to_narray
        expect(Numo::NArray.array_type(y)).to eq(Numo::DFloat)
        expect(y).to eq(Numo::DFloat.ones(2, 3))

        # TODO: float16

        x = MXNet::NDArray.ones([2, 3], dtype: :uint8)
        y = x.to_narray
        expect(Numo::NArray.array_type(y)).to eq(Numo::UInt8)
        expect(y).to eq(Numo::UInt8.ones(2, 3))

        x = MXNet::NDArray.ones([2, 3], dtype: :int32)
        y = x.to_narray
        expect(Numo::NArray.array_type(y)).to eq(Numo::Int32)
        expect(y).to eq(Numo::Int32.ones(2, 3))
      end
    end
  end
end
