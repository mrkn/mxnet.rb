require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Data::SimpleDataset do
  let(:dataset) do
    MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3, 4])
  end

  describe '#length' do
    specify do
      expect(MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3]).length).to eq(3)
      expect(MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3, 4, 5]).length).to eq(5)
    end
  end

  describe '#[]' do
    specify do
      expect(dataset[0]).to eq(1)
      expect(dataset[1]).to eq(2)
      expect(dataset[-1]).to eq(4)
    end
  end

  describe '#transform' do
    specify do
      res = dataset.transform {|x| x * 10 }
      expect(res).to be_a(MXNet::Gluon::Data::LazyTransformDataset)
      expect(res.length).to eq(4)
      expect(res[0]).to eq(10)
      expect(res[1]).to eq(20)
      expect(res[-1]).to eq(40)
    end

    context 'with lazy: false' do
      specify do
        res = dataset.transform(lazy: false) {|x| x * 10 }
        expect(res).to be_a(MXNet::Gluon::Data::SimpleDataset)
        expect(res.length).to eq(4)
        expect(res[0]).to eq(10)
        expect(res[1]).to eq(20)
        expect(res[-1]).to eq(40)
      end
    end
  end

  describe '#transform_first' do
    context 'with lazy: false' do
      specify do
        dataset = MXNet::Gluon::Data::SimpleDataset.new([[1, 2], [3, 4], [5, 6], [7, 8]])
        res = dataset.transform_first(lazy: false) {|x| x * 10 }
        expect(res).to be_a(MXNet::Gluon::Data::SimpleDataset)
        expect(res.length).to eq(4)
        expect(res[0]).to eq([10, 2])
        expect(res[1]).to eq([30, 4])
        expect(res[-1]).to eq([70, 8])
      end
    end
  end
end
