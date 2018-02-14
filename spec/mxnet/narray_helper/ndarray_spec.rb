require 'spec_helper'
require 'numo/narray'

module MXNet
  ::RSpec.describe NDArray do
    describe '#[]=' do
      xspecify do
        nd = MXNet::NDArray.arange(20, dtype: :float32).reshape([4, 5])
        nary = Numo::Int32.new(2, 3).fill(100)
        nd[1..2, 1..3] = nary # TODO: support this
        expect(nd.to_a).to eq([ 0.0,   1.0,   2.0,   3.0,  4.0,
                                5.0, 100.0, 100.0, 100.0,  9.0,
                               10.0, 100.0, 100.0, 100.0, 14.0,
                               15.0,  16.0,  17.0,  18.0, 19.0])
      end
    end

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

  ::RSpec.describe '.NDArray' do
    context 'with a Numo::Int8' do
      specify do
        x = Numo::Int8.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))

        y = MXNet::NDArray(x, dtype: :int8)
        expect(y.dtype).to eq(MXNet::DType.name2id(:int8))
        expect(y.to_narray).to eq(x)
      end
    end

    context 'with a Numo::Int16' do
      specify do
        x = Numo::Int32.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))
      end
    end

    context 'with a Numo::Int32' do
      specify do
        x = Numo::Int32.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))

        y = MXNet::NDArray(x, dtype: :int32)
        expect(y.dtype).to eq(MXNet::DType.name2id(:int32))
        expect(y.to_narray).to eq(x)
      end
    end

    context 'with a Numo::Int64' do
      specify do
        x = Numo::Int64.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))

        y = MXNet::NDArray(x, dtype: :int64)
        expect(y.dtype).to eq(MXNet::DType.name2id(:int64))
        expect(y.to_narray).to eq(x)
      end
    end

    context 'with a Numo::UInt8' do
      specify do
        x = Numo::UInt8.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))

        y = MXNet::NDArray(x, dtype: :uint8)
        expect(y.dtype).to eq(MXNet::DType.name2id(:uint8))
        expect(y.to_narray).to eq(x)
      end
    end

    context 'with a Numo::UInt16' do
      specify do
        x = Numo::UInt16.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))
      end
    end

    context 'with a Numo::UInt32' do
      specify do
        x = Numo::UInt32.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))
      end
    end

    context 'with a Numo::UInt64' do
      specify do
        x = Numo::UInt64.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))
      end
    end

    context 'with a Numo::SFloat' do
      specify do
        x = Numo::SFloat.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(x)
      end
    end

    context 'with a Numo::DFloat' do
      specify do
        x = Numo::DFloat.ones(2, 3)
        y = MXNet::NDArray(x)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float32))
        expect(y.to_narray).to eq(Numo::SFloat.cast(x))

        y = MXNet::NDArray(x, dtype: :float64)
        expect(y.dtype).to eq(MXNet::DType.name2id(:float64))
        expect(y.to_narray).to eq(x)
      end
    end
  end
end
