require 'spec_helper'

module MXNet
  ::RSpec.describe NDArray do
    describe '.new' do
      specify do
        expect { MXNet::NDArray.new }.to raise_error(NoMethodError)
      end
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
        expect { x.reshape({1 => 2}) }.to raise_error(TypeError, /no implicit conversion of Hash into Array/)
      end
    end

    describe ".reshape_like" do
      specify do
        x = MXNet::NDArray.zeros([2, 6])
        y = MXNet::NDArray.ones([3, 4])
        z = MXNet::NDArray.reshape_like(x, y)
        expect(z.shape).to eq([3, 4])
        expect(z.reshape([12]).to_a).to be_all {|x| x == 0.0 }
      end
    end

    describe '#dtype' do
      specify do
        expect(MXNet::NDArray.empty([1, 2]).dtype).to eq(:float32)
        expect(MXNet::NDArray.empty([1, 2], dtype: :float32).dtype).to eq(:float32)
        expect(MXNet::NDArray.empty([1, 2], dtype: :float64).dtype).to eq(:float64)
        expect(MXNet::NDArray.empty([1, 2], dtype: :float16).dtype).to eq(:float16)
        expect(MXNet::NDArray.empty([1, 2], dtype: :uint8).dtype).to eq(:uint8)
        expect(MXNet::NDArray.empty([1, 2], dtype: :int8).dtype).to eq(:int8)
        expect(MXNet::NDArray.empty([1, 2], dtype: :int32).dtype).to eq(:int32)
        expect(MXNet::NDArray.empty([1, 2], dtype: :int64).dtype).to eq(:int64)
      end
    end


    describe '#stype and #to_stype' do
      specify do
        original = MXNet::NDArray.ones([1, 2]) 
        expect(original.stype).to eq(:default)

        casted = original.to_stype :csr
        expect(casted.stype).to eq(:csr)
        #ensure integrity of values
        expect(MXNet::NDArray.sum((original == casted), exclude: true).to_i).to eq(original.shape.reduce :*)
        
        casted = casted.to_stype :default
        expect(casted.stype).to eq(:default)
        #ensure integrity of values
        expect(MXNet::NDArray.sum((original == casted), exclude: true).to_i).to eq(original.shape.reduce :*)
        

        casted = original.to_stype :row_sparse
        expect(casted.stype).to eq(:row_sparse)
        #ensure integrity of values
        expect(MXNet::NDArray.sum((original == casted), exclude: true).to_i).to eq(original.shape.reduce :*)


        casted = casted.to_stype :default
        expect(casted.stype).to eq(:default)
        #ensure integrity of values
        expect(MXNet::NDArray.sum((original == casted), exclude: true).to_i).to eq(original.shape.reduce :*)
        
        csr_shape_lt_2 = MXNet::NDArray.empty([1])
        expect{csr_shape_lt_2.to_stype :csr }.to raise_error

        csr_shape_gt_2 = MXNet::NDArray.empty([1,2,3])
        expect{csr_shape_gt_2.to_stype :csr }.to raise_error



        row_sparse_shape_lt_2 = MXNet::NDArray.empty([1,2,3])
        expect{row_sparse_shape_lt_2.to_stype :row_sparse }.to_not raise_error

        row_sparse_shape_gt_2 = MXNet::NDArray.empty([1,2,3])
        expect{row_sparse_shape_gt_2.to_stype :row_sparse }.to_not raise_error


        
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

    describe "#T" do
      specify do
        x = MXNet::NDArray.arange(0, 6).reshape([2, 3])
        expect(x.to_narray).to eq([[0, 1, 2], [3, 4, 5]])
        expect(x.T.to_narray).to eq([[0, 3], [1, 4], [2, 5]])
      end
    end

    describe '#dup' do
      pending
    end

    describe '#as_scalar' do
      pending
    end

    describe '#as_type' do
      specify do
        x = MXNet::NDArray.zeros([2, 3], dtype: :float32)
        y = x.as_type(:int32)
        expect(y.dtype).to eq(:int32)
      end
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

    describe "#topk" do
      let(:x) { MXNet::NDArray.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]]) }

      specify "returns an index of the largest element on last axis" do
        expect(x.topk.to_narray).to eq([[2], [1]])
      end

      specify "returns the value of top-2 largest elements on last axis" do
        expect(x.topk(ret_typ: "value", k: 2).to_narray).to eq([[0.4, 0.3], [0.3, 0.2]])
      end

      specify "returns the value of top-2 smallest elements on last axis" do
        expect(x.topk(ret_typ: "value", k: 2, is_ascend: true).to_narray).to eq([[0.2, 0.3], [0.1, 0.2]])
      end

      specify "returns the value of top-2 largest elements on axis 0" do
        expect(x.topk(axis: 0, ret_typ: "value", k: 2).to_narray).to eq([[0.3, 0.3, 0.4], [0.1, 0.2, 0.2]])
      end

      specify "flattens and then returns list of both values and indices" do
        res = x.topk(ret_typ: "both", k: 2)
        expect(res.size).to eq(2)
        expect(res.first.to_narray).to eq([[0.4, 0.3], [0.3, 0.2]])
        expect(res.last.to_narray).to eq([[2, 0], [1, 2]])
      end
    end

    describe "#pick" do
      let(:x) { MXNet::NDArray.arange(1, 7).reshape([3, 2]) }

      specify "picks elements with specified indices along axis 0" do
        y = MXNet::NDArray.array([0, 1])
        expect(x.pick(y, axis: 0).to_narray).to eq([1, 4])
      end

      specify "picks elements with specified indices along axis 1" do
        y = MXNet::NDArray.array([0, 1, 0])
        expect(x.pick(y, axis: 1).to_narray).to eq([1, 4, 5])
      end

      specify "picks elements with specified indices along axis 1 and dims are maintained" do
        y = MXNet::NDArray.array([[1], [0], [2]])
        expect(x.pick(y, axis: 1, keepdims: true).to_narray).to eq([[2], [3], [6]])
      end
    end

    describe "#sort" do
      let(:x) { MXNet::NDArray.array([[1, 4], [3, 1]]) }

      specify "sorts along the last axis" do
        expect(x.sort.to_narray).to eq([[1, 4], [1, 3]])
      end

      specify "sorts along the first axis" do
        expect(x.sort(axis: 0).to_narray).to eq([[1, 1], [3, 4]])
      end

      specify "sort in a descend order" do
        #TODO: when using is_ascend: false it doesn't work
        expect(x.sort(is_ascend: 0).to_narray).to eq([[4, 1], [3, 1]])
      end

      specify "sort in a descend order (with Ruby boolean)" do
        #TODO: when using is_ascend: false it doesn't work
        pending "is_ascend should work for false"
        expect(x.sort(is_ascend: false).to_narray).to eq([[4, 1], [3, 1]])
      end
    end

    describe '#tostype' do
      pending
    end

    describe '.argsort' do
      specify do
        x = MXNet::NDArray.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]])
        y = MXNet::NDArray.argsort(x)
        expect(y.reshape([-1]).to_a).to eq([1, 0, 2, 0, 2, 1])
        y = MXNet::NDArray.argsort(x, axis: 0)
        expect(y.reshape([-1]).to_a).to eq([1, 0, 1, 0, 1, 0])
        y = MXNet::NDArray.argsort(x, axis: 0, is_ascend: 'false')
        expect(y.reshape([-1]).to_a).to eq([0, 1, 0, 1, 0, 1])
      end

      specify do
        pending 'is_ascend should work for false'
        x = MXNet::NDArray.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]])
        y = MXNet::NDArray.argsort(x, axis: 0, is_ascend: false)
        expect(y.reshape([-1]).to_a).to eq([0, 1, 0, 1, 0, 1])
      end
    end

    describe '#argsort' do
      specify do
        x = MXNet::NDArray.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]])
        y = x.argsort
        expect(y.reshape([-1]).to_a).to eq([1, 0, 2, 0, 2, 1])
        y = x.argsort(axis: 0)
        expect(y.reshape([-1]).to_a).to eq([1, 0, 1, 0, 1, 0])
        y = x.argsort(axis: 0, is_ascend: 'false')
        expect(y.reshape([-1]).to_a).to eq([0, 1, 0, 1, 0, 1])
      end

      specify do
        pending 'is_ascend should work for false'
        x = MXNet::NDArray.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]])
        y = x.argsort(axis: 0, is_ascend: false)
        expect(y.reshape([-1]).to_a).to eq([0, 1, 0, 1, 0, 1])
      end
    end

    describe '.argmax' do
      specify do
        x = MXNet::NDArray.array([[0, 1, 2], [3, 4, 5]])
        y = MXNet::NDArray.argmax(x, axis: 0)
        expect(y.reshape([-1]).to_a).to eq([1, 1, 1])
        y = MXNet::NDArray.argmax(x, axis: 1)
        expect(y.reshape([-1]).to_a).to eq([2, 2])
        y = MXNet::NDArray.argmax(x, axis: 1, keepdims: true)
        expect(y.shape).to eq([2, 1])
        expect(y.reshape([-1]).to_a).to eq([2, 2])
      end
    end

    describe '#argmax' do
      specify do
        x = MXNet::NDArray.array([[0, 1, 2], [3, 4, 5]])
        y = x.argmax(axis: 0)
        expect(y.reshape([-1]).to_a).to eq([1, 1, 1])
        y = x.argmax(axis: 1)
        expect(y.reshape([-1]).to_a).to eq([2, 2])
        y = x.argmax(axis: 1, keepdims: true)
        expect(y.shape).to eq([2, 1])
        expect(y.reshape([-1]).to_a).to eq([2, 2])
      end
    end

    describe "#argmax_channel" do
      specify do
        x = MXNet::NDArray.array([[0, 1, 2], [3, 4, 5], [8, 7, 6]])
        expect(x.argmax_channel.to_a).to eq([2, 2, 0])
      end
    end

    describe '.argmin' do
      specify do
        x = MXNet::NDArray.array([[0, 1, 2], [3, 4, 5]])
        y = MXNet::NDArray.argmin(x, axis: 0)
        expect(y.reshape([-1]).to_a).to eq([0, 0, 0])
        y = MXNet::NDArray.argmin(x, axis: 1)
        expect(y.reshape([-1]).to_a).to eq([0, 0])
        y = MXNet::NDArray.argmin(x, axis: 1, keepdims: true)
        expect(y.shape).to eq([2, 1])
        expect(y.reshape([-1]).to_a).to eq([0, 0])
      end
    end

    describe '#argmin' do
      specify do
        x = MXNet::NDArray.array([[0, 1, 2], [3, 4, 5]])
        y = x.argmin(axis: 0)
        expect(y.reshape([-1]).to_a).to eq([0, 0, 0])
        y = x.argmin(axis: 1)
        expect(y.reshape([-1]).to_a).to eq([0, 0])
        y = x.argmin(axis: 1, keepdims: true)
        expect(y.shape).to eq([2, 1])
        expect(y.reshape([-1]).to_a).to eq([0, 0])
      end
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

    describe ".ones_like" do
      specify do
        x = MXNet::NDArray.zeros([3, 4])
        z = MXNet::NDArray.ones_like(x)
        expect(z.shape).to eq([3, 4])
        expect(z.reshape([12]).to_a).to be_all {|x| x == 1.0 }
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

    describe ".zeros_like" do
      specify do
        x = MXNet::NDArray.ones([3, 4])
        z = MXNet::NDArray.zeros_like(x)
        expect(z.shape).to eq([3, 4])
        expect(z.reshape([12]).to_a).to be_all {|x| x == 0.0 }
      end
    end

    describe "#broadcast_axes" do
      # given x of shape (1,2,1)
      let(:x) { MXNet::NDArray.array([[[1], [2]]]) }

      specify "broadcast x on on axis 2" do
        expect(x.broadcast_axes(axis: 2, size: 3).to_narray).to eq([[[1, 1, 1], [2, 2, 2]]])
      end

      specify "broadcast x on on axes 0 and 2" do
        expect(x.broadcast_axes(axis: [0, 2], size: [2, 3]).to_narray).to eq(
          [[[1, 1, 1], [2, 2, 2]],
           [[1, 1, 1], [2, 2, 2]]]
        )
      end
    end
    describe "#repeat" do
      let(:x) { MXNet::NDArray.array([[1, 2], [3, 4]]) }

      specify "repeat along axis 0" do
        expect(x.repeat(repeats: 2, axis: 0).to_narray).to eq(
          [[1, 2],
           [1, 2],
           [3, 4],
           [3, 4]]
        )
      end

      specify "repeat along axis 1" do
        expect(x.repeat(repeats: 2, axis: -1).to_narray).to eq(
          [[1, 1, 2, 2],
           [3, 3, 4, 4]]
        )
      end

      specify "repeat along the last axis" do
        expect(x.repeat(repeats: 2, axis: -1).to_narray).to eq(
          [[1, 1, 2, 2],
           [3, 3, 4, 4]]
        )
      end
    end

    describe "#pad" do
      let(:x) do
        MXNet::NDArray.array(
          [[[[1, 2, 3],
            [4, 5, 6]],
            [[7, 8, 9],
            [10, 11, 12]]],
          [[[11, 12, 13],
            [14, 15, 16]],
            [[17, 18, 19],
            [20, 21, 22]]]]
        )
      end

      specify "pads using the edge values of the input array" do
        expect(x.pad(mode: "edge", pad_width: [0, 0, 0, 0, 1, 1, 1, 1]).to_narray).to eq(
          [[[[1, 1, 2, 3, 3],
             [1, 1, 2, 3, 3],
             [4, 4, 5, 6, 6],
             [4, 4, 5, 6, 6]],

            [[7, 7, 8, 9, 9],
             [7, 7, 8, 9, 9],
             [10, 10, 11, 12, 12],
             [10, 10, 11, 12, 12]]],

           [[[11, 11, 12, 13, 13],
             [11, 11, 12, 13, 13],
             [14, 14, 15, 16, 16],
             [14, 14, 15, 16, 16]],

            [[17, 17, 18, 19, 19],
             [17, 17, 18, 19, 19],
             [20, 20, 21, 22, 22],
             [20, 20, 21, 22, 22]]]]
        )
      end

      specify "pads using a constant value" do
        expect(x.pad(mode: "constant", constant_value: 0, pad_width: [0, 0, 0, 0, 1, 1, 1, 1]).to_narray).to eq(
          [[[[0, 0, 0, 0, 0],
             [0, 1, 2, 3, 0],
             [0, 4, 5, 6, 0],
             [0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [0, 7, 8, 9, 0],
             [0, 10, 11, 12, 0],
             [0, 0, 0, 0, 0]]],

           [[[0, 0, 0, 0, 0],
             [0, 11, 12, 13, 0],
             [0, 14, 15, 16, 0],
             [0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [0, 17, 18, 19, 0],
             [0, 20, 21, 22, 0],
             [0, 0, 0, 0, 0]]]]
        )
      end
    end

    describe "#swapaxes" do
      specify "interchange axis 0 and 1 of a 1,3 array" do
        x = MXNet::NDArray.array([[1, 2, 3]])
        expect(x.swapaxes(dim1: 0, dim2: 1).to_narray).to eq([[1], [2], [3]])
      end

      specify "interchange axis 0 and 2 of a 2,2,2 array" do
        x = MXNet::NDArray.array(
          [[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]]
        )
        expect(x.swapaxes(dim1: 0, dim2: 2).to_narray).to eq(
          [[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]]
        )
      end
    end

    describe "#split" do
      let(:x) { MXNet::NDArray.array([[[1], [2]], [[3], [4]], [[5], [6]]]) }

      specify "split a 3,2,1 array into two 3,1,1 arrays" do
        expect(x.shape).to eq([3, 2, 1])
        y = x.split(axis: 1, num_outputs: 2)
        expect(y.size).to eq(2)
        expect(y[0].shape).to eq([3, 1, 1])
        expect(y[0].to_narray).to eq ([[[1]], [[3]], [[5]]])
        expect(y[1].to_narray).to eq ([[[2]], [[4]], [[6]]])
      end

      specify "split a 3,2,1 array into three 1,2,1 arrays" do
        y = x.split(axis: 0, num_outputs: 3)
        expect(y.size).to eq(3)
        expect(y[0].shape).to eq([1, 2, 1])
        expect(y[0].to_narray).to eq([[[1], [2]]])
        expect(y[1].to_narray).to eq([[[3], [4]]])
        expect(y[2].to_narray).to eq([[[5], [6]]])
      end

      specify "squeeze axis with length 1 from the shapes of the 3 output arrays with shape (2, 1)" do
        y = x.split(axis: 0, num_outputs: 3, squeeze_axis: 1)
        expect(y.size).to eq(3)
        expect(y[0].shape).to eq([2, 1])
        expect(y[0].to_narray).to eq([[1], [2]])
        expect(y[1].to_narray).to eq([[3], [4]])
        expect(y[2].to_narray).to eq([[5], [6]])
      end
    end

    describe "#slice" do
      let(:x) { MXNet::NDArray.arange(1, 13).reshape([3, 4]) }

      specify do
        expect(x.slice(begin: [0, 1], end: [2, 4]).to_narray).to eq(
          [[2, 3, 4],
           [6, 7, 8]]
        )

        expect(x.slice(begin: [None, 0], end: [None, 3], step: [-1, 2]).to_narray).to eq(
          [[9, 11], [5, 7], [1, 3]]
        )
      end
    end

    describe "#slice_axis" do
      let(:x) { MXNet::NDArray.arange(1, 13).reshape([3, 4]) }

      specify "slice elements 1 and 2 along axis 0" do
        expect(x.slice_axis(axis: 0, begin: 1, end: 3).to_narray).to eq(
          [[5, 6, 7, 8],
           [9, 10, 11, 12]]
        )
      end

      specify "slice elements 0 and 1 along axis 1" do
        expect(x.slice_axis(axis: 1, begin: 0, end: 2).to_narray).to eq(
          [[1, 2], [5, 6], [9, 10]]
        )
      end

      specify "slice elements 2 and 3 along axis 1 counting from the end" do
        expect(x.slice_axis(axis: 1, begin: -3, end: -1).to_narray).to eq(
          [[2, 3], [6, 7], [10, 11]]
        )
      end
    end

    describe "#slice_like" do
      let(:x) { MXNet::NDArray.arange(1, 13).reshape([3, 4]) }

      let(:y) { MXNet::NDArray.zeros([2, 3]) }

      specify "slice along all axes" do
        expect(x.slice_like(y).to_narray).to eq(
          [[1, 2, 3],
           [5, 6, 7]]
        )
      end

      specify "slice along axis 0 and 1 (same as above)" do
        expect(x.slice_like(y, axes: [0, 1]).to_narray).to eq(
          [[1, 2, 3],
           [5, 6, 7]]
        )
      end

      specify "slice along axis 0" do
        expect(x.slice_like(y, axes: [0]).to_narray).to eq(
          [[1, 2, 3, 4],
           [5, 6, 7, 8]]
        )
      end

      specify "slice along axis -1" do
        expect(x.slice_like(y, axes: [-1]).to_narray).to eq(
          [[1, 2, 3],
           [5, 6, 7],
           [9, 10, 11]]
        )
      end
    end

    describe "#take" do
      specify "take the second element along the first axis" do
        x = MXNet::NDArray.array([4, 5, 6])
        indices = MXNet::NDArray.array([1]) # can not use a Ruby Array for indices
        expect(x.take(indices).to_narray).to eq([5])
      end

      specify "take rows 0 and 1, then 1 and 2 along axis 0" do
        x = MXNet::NDArray.array([[1, 2], [3, 4], [5, 6]])
        indices = MXNet::NDArray.array([[0, 1], [1, 2]]) # can not use a Ruby Array for indices
        expect(x.take(indices).to_narray).to eq(
          [[[1, 2], [3, 4]],
           [[3, 4], [5, 6]]]
        )
      end
    end

    describe ".one_hot" do
      specify "one_hot with default on and off values and depth of 3" do
        indices = MXNet::NDArray.array([1, 0, 2, 0]) # can not use a Ruby Array for indices
        expect(MXNet::NDArray.one_hot(indices, depth: 3).to_narray).to eq(
          [[0, 1, 0],
           [1, 0, 0],
           [0, 0, 1],
           [1, 0, 0]]
        )
      end

      specify "one_hot with specific on and off values and dtype int32" do
        indices = MXNet::NDArray.array([1, 0, 2, 0]) # can not use a Ruby Array for indices
        expect(MXNet::NDArray.one_hot(indices, depth: 3, on_value: 8, off_value: 1, dtype: :int32).to_narray).to eq(
          [[1, 8, 1],
           [8, 1, 1],
           [1, 1, 8],
           [8, 1, 1]]
        )
      end

      specify "one_hot with multi-level indices" do
        indices = MXNet::NDArray.array([[1, 0], [1, 0], [2, 0]]) # can not use a Ruby Array for indices
        expect(MXNet::NDArray.one_hot(indices, depth: 3).to_narray).to eq(
          [[[0, 1, 0],
            [1, 0, 0]],
           [[0, 1, 0],
            [1, 0, 0]],
           [[0, 0, 1],
            [1, 0, 0]]]
        )
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

    describe '#moveaxis' do
      specify do
        pending "moveaxis is unknown from Ops delegator. Why ?"
        x = MXNet::NDArray.array([[1, 2, 3], [4, 5, 6]])
        expect(x.moveaxis(0,1).to_narray).to eq([[1,2], [3,4], [5,6]])
      end
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

    describe "#clip" do
      specify do
        x = MXNet::NDArray.arange(0, 10)
        expect(x.clip(a_min: 1, a_max: 8).to_narray).to eq([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
      end
    end

    describe "#sign" do
      specify do
        x = MXNet::NDArray.array([-2, 0, 3])
        expect(x.sign.to_narray).to eq([-1, 0, 1])
      end
    end

    describe "#flatten" do
      specify do
        x = MXNet::NDArray.arange(1, 19).reshape([2, 3, 3])
        expect(x.flatten.to_narray).to eq(
          [[1, 2, 3, 4, 5, 6, 7, 8, 9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18]]
        )
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

    describe '#wait_to_read' do
      specify do
        x = MXNet::NDArray.array([1, 2, 3])
        y = MXNet::NDArray.dot(x, x)
        expect { y.wait_to_read }.not_to raise_error
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
