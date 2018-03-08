require 'spec_helper'

module MXNet
  ::RSpec.describe NDArray do
    describe '#save and .load' do
      pending
    end

    describe '#[]' do
      context 'when the array is 1D' do
        specify do
          x = MXNet::NDArray.zeros([3])
          expect(x[0]).to be_a(MXNet::NDArray)
          expect(x[0].as_scalar).to eq(0.0)
          expect(x[1].as_scalar).to eq(0.0)
          expect(x[2].as_scalar).to eq(0.0)
          expect(x.shape).to eq([3])
        end

        specify do
          x = MXNet::NDArray.arange(10)
          expect(x[3...6]).to be_a(MXNet::NDArray)
          expect(x[3...6].shape).to eq([3])
          expect(x[3...6].to_a).to eq([3.0, 4.0, 5.0])
        end

        specify do
          x = MXNet::NDArray.arange(11)
          expect(x[(1...9).step(2)]).to be_a(MXNet::NDArray)
          expect(x[(1...9).step(2)].shape).to eq([4])
          expect(x[(1...9).step(2)].to_a).to eq([1.0, 3.0, 5.0, 7.0])
          expect(x[(1...10).step(2)].to_a).to eq([1.0, 3.0, 5.0, 7.0, 9.0])
          expect(x[(9...1).step(-2)].to_a).to eq([9.0, 7.0, 5.0, 3.0])
        end
      end

      context 'when the array is 2D' do
        specify do
          x = MXNet::NDArray.zeros([2,3])
          expect(x[0]).to be_a(MXNet::NDArray)
          expect(x[1]).to be_a(MXNet::NDArray)
          expect(x[0][0].as_scalar).to eq(0.0)
          expect(x[1][2].as_scalar).to eq(0.0)
          expect(x[1][1].as_scalar).to eq(0.0)
          expect(x[0].shape).to eq([3])
          expect(x[1].shape).to eq([3])
        end

        specify do
          x = MXNet::NDArray.ones([6, 100])
          expect(x[0..-1, 0]).to be_a(MXNet::NDArray)
          expect(x[0..-1, 1]).to be_a(MXNet::NDArray)
          expect(x[0..-1, 0].shape).to eq([6])
        end
      end

      context 'when the array is 3D' do
        specify do
          x = MXNet::NDArray.zeros([2,3,4])
          expect(x[0]).to be_a(MXNet::NDArray)
          expect(x[1]).to be_a(MXNet::NDArray)
          expect(x[0][0][0].as_scalar).to eq(0.0)
          expect(x[1][2][3].as_scalar).to eq(0.0)
          expect(x[1][1][1].as_scalar).to eq(0.0)
          expect(x[0].shape).to eq([3,4])
          expect(x[0][1].shape).to eq([4])
        end
      end
    end

    describe '#[]=' do
      context 'when the array  is 1D' do
        specify do
          x = MXNet::NDArray.zeros([3])
          x[1] = 10.0
          expect(x.to_a).to eq([0.0, 10.0, 0.0])
        end

        specify do
          x = MXNet::NDArray.zeros([5])
          x[1...3] = 10.0
          expect(x.to_a).to eq([0.0, 10.0, 10.0, 0.0, 0.0])
        end
      end

      context 'when the array is 2D' do
        specify do
          x = MXNet::NDArray.arange(0, 9).reshape([3, 3])
          x[1][1] = 100.0
          expect(x.reshape([9]).to_a).to eq([0.0, 1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0])
        end

        specify do
          x = MXNet::NDArray.arange(0, 9).reshape([3, 3])
          x[1][0..-1] = 200.0
          expect(x.reshape([9]).to_a).to eq([0.0, 1.0, 2.0, 200.0, 200.0, 200.0, 6.0, 7.0, 8.0])
        end

        specify do
          x = MXNet::NDArray.arange(0, 9).reshape([3, 3])
          x[1][0..-1] = MXNet::NDArray.arange(10, 13)
          expect(x.reshape([9]).to_a).to eq([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 6.0, 7.0, 8.0])
        end
      end
    end

    describe '#reshape' do
      specify do
        x = MXNet::NDArray.zeros([2, 3])
        expect(x.reshape([6]).shape).to eq([6])
        expect(x.reshape([6, 1]).shape).to eq([6, 1])
        expect(x.reshape([3, 2]).shape).to eq([3, 2])
        expect(x.reshape([2, 2]).shape).to eq([2, 2])
        expect { x.reshape([2, 2, 2]) }.to raise_error(MXNet::Error, /target shape size is larger current shape/)
      end
    end

    describe '#dtype' do
      specify do
        expect(MXNet::NDArray.empty([1, 2]).dtype).to eq(DType.name2id(:float32))
        expect(MXNet::NDArray.empty([1, 2], dtype: :float32).dtype).to eq(DType.name2id(:float32))
        expect(MXNet::NDArray.empty([1, 2], dtype: :float64).dtype).to eq(DType.name2id(:float64))
        expect(MXNet::NDArray.empty([1, 2], dtype: :float16).dtype).to eq(DType.name2id(:float16))
        expect(MXNet::NDArray.empty([1, 2], dtype: :uint8).dtype).to eq(DType.name2id(:uint8))
        expect(MXNet::NDArray.empty([1, 2], dtype: :int8).dtype).to eq(DType.name2id(:int8))
        expect(MXNet::NDArray.empty([1, 2], dtype: :int32).dtype).to eq(DType.name2id(:int32))
        expect(MXNet::NDArray.empty([1, 2], dtype: :int64).dtype).to eq(DType.name2id(:int64))
      end
    end

    describe '#ndim' do
      specify do
        x = MXNet::NDArray.empty([3, 2, 1, 4])
        expect(x.ndim).to eq(4)
      end
    end

    describe '#shape' do
      specify do
        x = MXNet::NDArray.empty([3, 2, 1, 4])
        expect(x.shape).to eq([3, 2, 1, 4])
      end
    end

    describe '#size' do
      specify do
        x = MXNet::NDArray.empty([3, 2, 4])
        expect(x.size).to eq(3*2*4)
      end
    end

    describe '#context' do
      specify do
        x = MXNet::NDArray.ones([2, 3])
        expect(x.context).to eq(MXNet.current_context)
      end

      specify do
        x = MXNet::NDArray.ones([2, 3], ctx: MXNet.cpu)
        expect(x.context).to eq(MXNet.cpu)
      end
    end

    describe '#dtype' do
      pending
    end

    describe '#stype' do
      pending
    end

    describe '#transpose' do
      specify do
        x = MXNet::NDArray.empty([3, 2, 1, 4])
        x_t = x.transpose
        expect(x_t.shape).to eq([4, 1, 2, 3])
      end

      specify do
        x = MXNet::NDArray.empty([3, 2, 1, 4])
        x_t = x.transpose(axes: [1, 2, 3, 0])
        expect(x_t.shape).to eq([2, 1, 4, 3])
      end
    end

    describe '#dup' do
      pending
    end

    describe '#as_in_context' do
      pending
    end

    describe '#grad' do
      pending
    end

    describe '#detach' do
      pending
    end

    describe '#backward' do
      pending
    end

    describe '#tostype' do
      pending
    end

    describe '#clip' do
      pending
    end

    describe '#sqrt' do
      pending
    end

    describe '#dot' do
      pending
    end

    describe '.ones' do
      specify do
        x = MXNet::NDArray.ones([2, 1, 3])
        expect(x).to be_a(MXNet::NDArray)
        expect(x.shape).to eq([2, 1, 3])
        expect(x.reshape([6]).to_a).to be_all {|x| x == 1.0 }
      end
    end

    describe '.zeros' do
      specify do
        x = MXNet::NDArray.zeros([2, 1, 3])
        expect(x).to be_a(MXNet::NDArray)
        expect(x.shape).to eq([2, 1, 3])
        expect(x.reshape([6]).to_a).to be_all {|x| x == 0.0 }
      end
    end

    describe '.empty' do
      specify do
        x = MXNet::NDArray.empty([2, 1, 3])
        expect(x).to be_a(MXNet::NDArray)
        expect(x.shape).to eq([2, 1, 3])
      end
    end

    describe '.arange' do
      specify do
        x = MXNet::NDArray.arange(3)
        expect(x.to_a).to eq([0.0, 1.0, 2.0])

        x = MXNet::NDArray.arange(2, 6)
        expect(x.to_a).to eq([2.0, 3.0, 4.0, 5.0])

        x = MXNet::NDArray.arange(2, 8, step: 2)
        expect(x.to_a).to eq([2.0, 4.0, 6.0])

        x = MXNet::NDArray.arange(2, 6, step: 1.5, repeat: 2)
        expect(x.to_a).to eq([2.0, 2.0, 3.5, 3.5, 5.0, 5.0])

        x = MXNet::NDArray.arange(2, 6, step: 2, repeat: 3, dtype: :int32)
        expect(x.to_a).to eq([2, 2, 2, 4, 4, 4])
      end
    end

    describe '.full' do
      pending
    end

    describe '.array' do
    end

    describe '#move_axis' do
      pending
    end

    describe '.arange' do
      pending
    end

    describe '#+' do
      describe 'ndarray + ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = MXNet::NDArray.ones([2, 3, 4])
          z = x + y
          expect(z.reshape([24]).to_a).to eq([2.0] * 24)
        end
      end

      describe 'ndarray + scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = x + 1
          expect(y.to_a).to eq([1, 2, 3])
        end
      end

      describe 'scalar + ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 1 + x
          expect(y.to_a).to eq([1, 2, 3])
        end
      end
    end

    describe '#-' do
      describe 'ndarray - ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = MXNet::NDArray.ones([2, 3, 4])
          z = x - y
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray - scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = x - 1
          expect(y.to_a).to eq([-1, 0, 1])
        end
      end

      describe 'scalar - ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 2 - x
          expect(y.to_a).to eq([2, 1, 0])
        end
      end
    end

    describe '#*' do
      describe 'ndarray * ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = x + x
          z = y * y
          expect(z.reshape([24]).to_a).to eq([4.0] * 24)
        end
      end

      describe 'ndarray * scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = x * 10
          expect(y.to_a).to eq([0, 10, 20])
        end
      end

      describe 'scalar * ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 10 * x
          expect(y.to_a).to eq([0, 10, 20])
        end
      end
    end

    describe '#/' do
      describe 'ndarray / ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x) / (x + x + x)
          expect(z.reshape([24]).to_a).to eq([2.0 / 3.0] * 24)
        end
      end

      describe 'ndarray / scalar' do
        specify do
          x = MXNet::NDArray.arange(3, dtype: :float64)
          y = x / 10
          expect(y.to_a).to eq([0, 0.1, 0.2])
        end
      end

      describe 'scalar / ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 12 / (x + 1)
          expect(y.to_a).to eq([12, 6, 4])
        end
      end
    end

    describe '#%' do
      describe 'ndarray % ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) % (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
        end
      end

      describe 'ndarray % scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = (x + 2) % 3
          expect(y.to_a).to eq([2, 0, 1])
        end
      end

      describe 'scalar % ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 11 % (x + 1)
          expect(y.to_a).to eq([0, 1, 2])
        end
      end
    end

    describe '#**' do
      describe 'ndarray ** ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) ** (x + x)
          expect(z.reshape([24]).to_a).to eq([9.0] * 24)
        end
      end

      describe 'ndarray ** scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = x ** 2
          expect(y.to_a).to eq([0, 1, 4])
        end
      end

      describe 'scalar ** ndarray' do
        specify do
          x = MXNet::NDArray.arange(3)
          y = 3 ** x
          expect(y.to_a).to eq([1, 3, 9])
        end
      end
    end

    describe '#maximum' do
      pending
    end

    describe '#minimum' do
      pending
    end

    describe '#==' do
      describe 'ndarray == ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) == (x + x)
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
          z = (x + x) == (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
        end
      end

      describe 'ndarray == scalar' do
        pending
      end

      describe 'scalar == ndarray' do
        pending
      end
    end

    describe '#!=' do
      describe 'ndarray != ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) != (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) != (x + x)
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray != scalar' do
        pending
      end

      describe 'scalar != ndarray' do
        pending
      end
    end

    describe '#>' do
      describe 'ndarray > ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) > (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) > (x + x)
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray > scalar' do
        pending
      end

      describe 'scalar > ndarray' do
        pending
      end
    end

    describe '#>=' do
      describe 'ndarray >= ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x + x) >= (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) >= (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = x >= (x + x)
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray >= scalar' do
        pending
      end

      describe 'scalar >= ndarray' do
        pending
      end
    end

    describe '#<' do
      describe 'ndarray < ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x) < (x + x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) < (x + x)
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray < scalar' do
        pending
      end

      describe 'scalar < ndarray' do
        pending
      end
    end

    describe '#<=' do
      describe 'ndarray <= ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          z = (x + x) <= (x + x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) <= (x + x)
          expect(z.reshape([24]).to_a).to eq([1.0] * 24)
          z = (x + x) <= x
          expect(z.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray <= scalar' do
        pending
      end

      describe 'scalar <= ndarray' do
        pending
      end
    end

    describe '#@+' do
      specify do
        x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
        z = +(x + x)
        expect(z.reshape([24]).to_a).to eq([2.0] * 24)
      end
    end

    describe '#@-' do
      specify do
        x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
        z = -(x + x)
        expect(z.reshape([24]).to_a).to eq([-2.0] * 24)
      end
    end

    describe '.concat' do
      pending
    end
  end
end
