require 'spec_helper'

module MXNet
  ::RSpec.describe NDArray do
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

        specify do
          x = MXNet::NDArray.ones([3, 3])
          x[0..-1, 1] = 10
          expect(x.reshape([9]).to_a).to eq([1.0, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 10.0, 1.0])
        end

        specify do
          x = MXNet::NDArray.zeros(3)
          x[1] += 1
          expect(x.to_a).to eq([0, 1, 0])
        end

        specify do
          x = MXNet::NDArray.ones([4, 4])
          y = MXNet::NDArray.zeros([2, 2])
          x[1..2, 1..2] = y
          expect(x.reshape([16]).to_a).to eq([1.0, 1.0, 1.0, 1.0,
                                              1.0, 0.0, 0.0, 1.0,
                                              1.0, 0.0, 0.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0])
        end

        xspecify do
          x = MXNet::NDArray.ones([4, 4])
          y = MXNet::NDArray.zeros([1])
          x[1..2, 1..2] = y
          expect(x.reshape([16]).to_a).to eq([1.0, 1.0, 1.0, 1.0,
                                              1.0, 0.0, 0.0, 1.0,
                                              1.0, 0.0, 0.0, 1.0,
                                              1.0, 1.0, 1.0, 1.0])
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

    describe '.abs' do
      specify do
        x = MXNet::NDArray.array([-1, 2, -3])
        z = MXNet::NDArray.abs(x)
        expect(z.to_a).to eq([1, 2, 3])
      end
    end

    describe '#abs' do
      specify do
        x = MXNet::NDArray.array([-1, 2, -3])
        z = x.abs
        expect(z.to_a).to eq([1, 2, 3])
      end
    end

    describe '.max' do
      specify do
        x = MXNet::NDArray.array([[1, 2, 3, 4], [6, 5, 4, 3]])
        z = MXNet::NDArray.max(x)
        expect(z.to_a).to eq([6])
        z = MXNet::NDArray.max(x, axis: 0)
        expect(z.to_a).to eq([6, 5, 4, 4])
      end
    end

    describe '#max' do
      specify do
        x = MXNet::NDArray.array([[1, 2, 3, 4], [6, 5, 4, 3]])
        z = x.max
        expect(z.to_a).to eq([6])
        z = x.max(axis: 0)
        expect(z.to_a).to eq([6, 5, 4, 4])
      end
    end

    describe '.min' do
      specify do
        x = MXNet::NDArray.array([[4, 5, 2, 1], [6, 3, 0, 2]])
        z = MXNet::NDArray.min(x)
        expect(z.to_a).to eq([0])
        z = MXNet::NDArray.min(x, axis: 0)
        expect(z.to_a).to eq([4, 3, 0, 1])
      end
    end

    describe '#min' do
      specify do
        x = MXNet::NDArray.array([[4, 5, 2, 1], [6, 3, 0, 2]])
        z = x.min
        expect(z.to_a).to eq([0])
        z = x.min(axis: 0)
        expect(z.to_a).to eq([4, 3, 0, 1])
      end
    end

    describe '.sqrt' do
      specify do
        x = MXNet::NDArray.array([4, 9, 16])
        z = MXNet::NDArray.sqrt(x)
        expect(z.to_a).to eq([2, 3, 4])
      end
    end

    describe '#sqrt' do
      specify do
        x = MXNet::NDArray.array([4, 9, 16])
        z = x.sqrt
        expect(z.to_a).to eq([2, 3, 4])
      end
    end


    describe '.square' do
      specify do
        x = MXNet::NDArray.array([2, 3, 4])
        z = MXNet::NDArray.square(x)
        expect(z.to_a).to eq([4, 9, 16])
      end
    end

    describe '#square' do
      specify do
        x = MXNet::NDArray.array([2, 3, 4])
        z = x.square
        expect(z.to_a).to eq([4, 9, 16])
      end
    end

    describe '.dot' do
      specify do
        x = MXNet::NDArray.array([2,3,4]).reshape([1,3])
        y = MXNet::NDArray.array([7,6,5]).reshape([3,1])
        z = MXNet::NDArray.dot(x,y)
        expect(z.shape).to eq([1,1])
        expect(z[0].to_a).to eq([52])
      end

      specify do
        x = MXNet::NDArray.array([0,1,2,3,4,5,6,7]).reshape([2,2,2])
        y = MXNet::NDArray.array([7,6,5,4,3,2,1,0]).reshape([2,2,2])
        z = MXNet::NDArray.dot(x,y)
        expect(z.shape).to eq([2,2,2,2])
        expect(z.to_narray).to eq(
          [[[[3, 2],  [1, 0]],
           [[23, 18], [13, 8]]],
          [[[43, 34], [25, 16]],
           [[63, 50],  [37, 24]]]]
          )
      end
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
      specify do
        x = MXNet::NDArray.full([3], 3.14, dtype: :float64)
        expect(x.to_a).to eq([3.14, 3.14, 3.14])
      end
    end

    describe '.array' do
      specify do
        x = MXNet::NDArray.array([[1, 2], [3, 4]])
        expect(x.to_narray).to eq(Numo::SFloat.new(2, 2).seq(1))
      end
    end

    describe '#move_axis' do
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

      describe 'ndarray.inplace + ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = MXNet::NDArray.ones([2, 3, 4])
          x.inplace + y
          expect(x.reshape([24]).to_a).to eq([2.0] * 24)
        end
      end

      describe 'ndarray.inplace + scalar' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          x.inplace + 1
          expect(x.reshape([24]).to_a).to eq([2.0] * 24)
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

      describe 'ndarray.inplace - ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = MXNet::NDArray.ones([2, 3, 4])
          x.inplace - y
          expect(x.reshape([24]).to_a).to eq([0.0] * 24)
        end
      end

      describe 'ndarray.inplace - scalar' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          x.inplace - 1
          expect(x.reshape([24]).to_a).to eq([0.0] * 24)
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

      describe 'ndarray.inplace * ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4])
          y = x + x
          y.inplace * y
          expect(y.reshape([24]).to_a).to eq([4.0] * 24)
        end
      end

      describe 'ndarray.inplace * scalar' do
        specify do
          x = MXNet::NDArray.arange(3)
          x.inplace * 10
          expect(x.to_a).to eq([0, 10, 20])
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

      describe 'ndarray.inplace / ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          y = x + x
          y.inplace / (x + x + x)
          expect(y.reshape([24]).to_a).to eq([2.0 / 3.0] * 24)
        end
      end

      describe 'ndarray.inplace / scalar' do
        specify do
          x = MXNet::NDArray.arange(3, dtype: :float64)
          x.inplace / 10
          expect(x.to_a).to eq([0, 0.1, 0.2])
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

      describe 'ndarray.inplace / ndarray' do
        specify do
          x = MXNet::NDArray.ones([2, 3, 4], dtype: :float64)
          y = x + x + x
          y.inplace % (x + x)
          expect(y.reshape([24]).to_a).to eq([1.0] * 24)
        end
      end

      describe 'ndarray.inplace % scalar' do
        specify do
          x = MXNet::NDArray.arange(3, dtype: :float64)
          x.inplace + 2
          x.inplace % 3
          expect(x.to_a).to eq([2, 0, 1])
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

    describe '#argmax' do
      specify do
        x = MXNet::NDArray.arange(0,6).reshape([2,3]) # [[ 0,  1,  2], [ 3,  4,  5]]
        expect(MXNet::NDArray.argmax(x, axis:0).to_narray).to eq([1, 1, 1])
        expect(MXNet::NDArray.argmax(x, axis:1).to_narray).to eq([2, 2])
      end
    end

    describe '#argmin' do
      specify do
        x = MXNet::NDArray.arange(0,6).reshape([2,3]) # [[ 0,  1,  2], [ 3,  4,  5]]
        expect(MXNet::NDArray.argmin(x, axis:0).to_narray).to eq([0, 0, 0])
        expect(MXNet::NDArray.argmin(x, axis:1).to_narray).to eq([0, 0])
      end
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

    describe '#to_i' do
      specify do
        x = MXNet::NDArray.full([1], 3.14, dtype: :float64)
        expect(x.to_i).to eq(3)
      end
    end

    describe '#to_f' do
      specify do
        x = MXNet::NDArray.full([1], 3.14, dtype: :float64)
        expect(x.to_f).to eq(3.14)
      end
    end

    describe '#tile' do
      specify do
        # x = [[1,2], [3,4]] as in mxnet.ndarray.tile documentation
        x = MXNet::NDArray.arange(1,5).reshape([2,2])
        # case n=d
        y = x.tile(reps: [2,3])
        expect(y.to_narray).to eq(
          [[1, 2, 1, 2, 1, 2],
           [3, 4, 3, 4, 3, 4],
           [1, 2, 1, 2, 1, 2],
           [3, 4, 3, 4, 3, 4]]
        )

        # case n > d
        y = x.tile(reps: [2,]) # same as [1,2]
        expect(y.to_narray).to eq(
          [[1, 2, 1, 2],
           [3, 4, 3, 4]]
        )

        #case n < d
        y = x.tile(reps: [2,2,3])
        expect(y.to_narray).to eq(
          [[[1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]],
           [[1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
             [3, 4, 3, 4, 3, 4]]]
        )
      end
    end

    describe '.stack' do
      specify do
        x = MXNet::NDArray.array([1, 2])
        y = MXNet::NDArray.array([3, 4])

        z = MXNet::NDArray.stack(x, y)
        expect(z.to_narray).to eq([[1, 2], [3, 4]])
        z = MXNet::NDArray.stack(x, y, axis: 0)
        expect(z.to_narray).to eq([[1, 2], [3, 4]])
        z = MXNet::NDArray.stack(x, y, axis: 1)
        expect(z.to_narray).to eq([[1, 3], [2, 4]])
      end
    end

    describe '#attach_grad' do
      context 'default stype' do
        specify do
          x = MXNet::NDArray.zeros([10, 10])
          expect(x.grad).to eq(nil)
          x.attach_grad
          MXNet::Autograd.record do
            y = x * 2
            expect(y.grad).to eq(nil)
            y.backward(out_grad: MXNet::NDArray.ones_like(y))
          end
          expect(x.grad.reshape([100]).to_a).to be_all {|a| a == 2.0 }
        end
      end
    end

    describe '.maximum' do
      specify do
        x = MXNet::NDArray.ones([2,3])
        y = MXNet::NDArray.arange(2).reshape([2,1])
        z = MXNet::NDArray.arange(2).reshape([1,2])
        expect(MXNet::NDArray.maximum(x,2).to_narray).to eq(
          [[2, 2, 2],
           [2, 2, 2]]
        )
        expect(MXNet::NDArray.maximum(x,y).to_narray).to eq(
          [[1, 1, 1],
           [1, 1, 1]]
        )
        expect(MXNet::NDArray.maximum(y,z).to_narray).to eq(
          [[0, 1],
           [1, 1]]
        )
      end
    end

    describe '.minimum' do
      specify do
        x = MXNet::NDArray.ones([2,3])
        y = MXNet::NDArray.arange(2).reshape([2,1])
        z = MXNet::NDArray.arange(2).reshape([1,2])
        expect(MXNet::NDArray.minimum(x,2).to_narray).to eq(
           [[1, 1, 1],
            [1, 1, 1]]
        )
        expect(MXNet::NDArray.minimum(x,y).to_narray).to eq(
          [[0, 0, 0],
           [1, 1, 1]]
        )
        expect(MXNet::NDArray.minimum(y,z).to_narray).to eq(
          [[0, 0],
           [0, 1]]
        )
      end
    end

    describe '.concat' do
      specify do
        x = MXNet::NDArray.arange(4).reshape([2,2]) # [[0, 1], [2, 3]]
        y = MXNet::NDArray.arange(3,9).reshape([3,2]) # [[3, 4], [5, 6], [7, 8]]
        z = MXNet::NDArray.arange(9,15).reshape([3,2]) # [[9, 10], [11, 12], [13, 14]]
        expect(MXNet::NDArray.concat(x,y,z,dim: 0).to_narray).to eq(
          [[0, 1],
           [2, 3],
           [3, 4],
           [5, 6],
           [7, 8],
           [9, 10],
           [11, 12],
           [13, 14]]
       )
       expect(MXNet::NDArray.concat(y,z,dim: 1).to_narray).to eq(
        [[3, 4, 9, 10],
         [5, 6, 11, 12],
         [7, 8, 13, 14]]
       )
       # Cannot concatenate arrays along a dimension that is not the same for all arrays
       expect { MXNet::NDArray.concat(x,y,dim: 1) }.to raise_exception(MXNet::Error, /Incompatible input shape/)
      end
    end
  end
end
