require 'spec_helper'

module MXNet
  ::RSpec.describe Random do
    describe 'uniform' do
      specify do
        ary = NDArray.zeros([10])
        Random.uniform(2, 5, out: ary)
        expect(ary.to_a).to be_all {|x| 2 <= x && x < 5 }
      end

      specify do
        low = NDArray.zeros([2])
        low[0] = 0.0 # TODO
        low[1] = 2.0 # TODO
        high = NDArray.zeros([2])
        high[0] = 1.0 # TODO
        high[1] = 3.0 # TODO
        sample = Random.uniform(low, high)
        expect(sample.shape).to eq([2])
        expect(sample.to_a[0]).to be_between(0.0, 1.0).exclusive
        expect(sample.to_a[1]).to be_between(2.0, 3.0).exclusive
      end
    end

    describe 'normal' do
      specify do
        ary = NDArray.ones([10])
        Random.uniform(0, 1, out: ary)
        expect(ary.to_a).to be_any {|x| x != 1.0 }
      end
    end
  end
end
