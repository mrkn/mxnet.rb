require 'spec_helper'

module MXNet
  ::RSpec.describe NDArray do
    describe '#save and .load' do
      pending
    end

    describe '#[]=' do
      pending
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
      pending
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
      end
    end

    describe '.zeros' do
      specify do
        x = MXNet::NDArray.zeros([2, 1, 3])
        expect(x).to be_a(MXNet::NDArray)
        expect(x.shape).to eq([2, 1, 3])
      end
    end

    describe '.empty' do
      specify do
        x = MXNet::NDArray.empty([2, 1, 3])
        expect(x).to be_a(MXNet::NDArray)
        expect(x.shape).to eq([2, 1, 3])
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
      pending
    end

    describe '#-' do
      pending
    end

    describe '#*' do
      pending
    end

    describe '#/' do
      pending
    end

    describe '#%' do
      pending
    end

    describe '#**' do
      pending
    end

    describe '#maximum' do
      pending
    end

    describe '#minimum' do
      pending
    end

    describe '#==' do
      pending
    end

    describe '#!=' do
      pending
    end

    describe '#>' do
      pending
    end

    describe '#>=' do
      pending
    end

    describe '#<' do
      pending
    end

    describe '#<=' do
      pending
    end

    describe '#@-' do
      pending
    end

    describe '.concat' do
      pending
    end
  end
end
