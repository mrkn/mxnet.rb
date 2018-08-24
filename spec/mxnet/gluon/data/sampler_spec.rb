require 'spec_helper'
require 'mxnet/gluon/data/sampler'

RSpec.describe MXNet::Gluon::Data::SequentialSampler do
  let(:sampler) do
    described_class.new(100)
  end

  describe '#length' do
    it 'is 100' do
      expect(sampler.length).to eq(100)
    end
  end

  describe '#each' do
    it 'results in an array from 0 to 99 in sequence' do
      expect(sampler.each.to_a).to eq((0..99).to_a)
    end
  end
end

RSpec.describe MXNet::Gluon::Data::RandomSampler do
  let(:sampler) do
    described_class.new(100)
  end

  describe '#length' do
    it 'is 100' do
      expect(sampler.length).to eq(100)
    end
  end

  describe '#each' do
    it 'results in an array from 0 to 99 out of sequence' do
      expect(sampler.each.to_a).not_to eq((0..99).to_a)
    end
  end
end

RSpec.describe MXNet::Gluon::Data::BatchSampler do
  context 'with `last_batch = :keep`' do
    let(:sampler) do
      sampler = MXNet::Gluon::Data.SequentialSampler(10)
      described_class.new(sampler, 3, :keep)
    end

    describe '#length' do
      it 'is 4 (mini-batches)' do
        expect(sampler.length).to eq(4)
      end
    end

    describe '#each' do
      it 'results in a final partial batch' do
        expect(sampler.each.to_a).to eq([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
      end
    end
  end

  context 'with `last_batch = :discard`' do
    let(:sampler) do
      sampler = MXNet::Gluon::Data.SequentialSampler(10)
      described_class.new(sampler, 3, :discard)
    end

    describe '#length' do
      it 'is 3 (mini-batches)' do
        expect(sampler.length).to eq(3)
      end
    end

    describe '#each' do
      it 'results in an omitted final batch' do
        expect(sampler.each.to_a).to eq([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
      end
    end
  end

  context 'with `last_batch = :rollover`' do
    let(:sampler) do
      sampler = MXNet::Gluon::Data.SequentialSampler(10)
      described_class.new(sampler, 3, :rollover).tap do |sampler|
        2.times { sampler.to_a }
      end
    end

    describe '#length' do
      it 'is 3 (mini-batches)' do
        expect(sampler.length).to eq(4)
      end
    end

    describe '#each' do
      it 'results in a rolled-over final batch' do
        expect(sampler.each.to_a).to eq([[8, 9, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
      end
    end
  end
end
