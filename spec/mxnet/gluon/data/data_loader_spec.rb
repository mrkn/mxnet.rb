require 'spec_helper'
require 'mxnet/gluon/data/data_loader'

RSpec.describe MXNet::Gluon::Data::DataLoader do
  let(:dataset) do
    (0..99).to_a
  end
  let(:shuffle) do
    false
  end
  let(:batch_size) do
    3
  end
  let(:last_batch) do
    :keep
  end
  let(:loader) do
    described_class.new(dataset, shuffle: shuffle, batch_size: batch_size, last_batch: last_batch)
  end

  describe '#length' do
    it 'is 34 (mini-batches)' do
      expect(loader.length).to eq(34)
    end
  end

  describe '#each' do
    it 'results in an array of slices' do
      expect(loader.each.map(&:to_a)).to eq((0..99).each_slice(3).to_a)
    end
  end

  context 'with `shuffle: true`' do
    let(:shuffle) do
      true
    end

    describe '#length' do
      it 'is 4 (mini-batches)' do
        expect(loader.length).to eq(34)
      end
    end

    describe '#each' do
      it 'results in an array out of sequence' do
        expect(loader.each.map(&:to_a)).not_to eq((0..99).each_slice(3).to_a)
      end
    end
  end

  context 'with `batch_size: 10`' do
    let(:batch_size) do
      10
    end

    describe '#length' do
      it 'is 10 (mini-batches)' do
        expect(loader.length).to eq(10)
      end
    end

    describe '#each' do
      it 'results in an array of slices' do
        expect(loader.each.map(&:to_a)).to eq((0..99).each_slice(10).to_a)
      end
    end
  end

  context 'with `last_batch: :discard`' do
    let(:last_batch) do
      :discard
    end

    describe '#length' do
      it 'is 33 (mini-batches)' do
        expect(loader.length).to eq(33)
      end
    end

    describe '#each' do
      it 'results in an array of slices' do
        expect(loader.each.map(&:to_a)).to eq((0..98).each_slice(3).to_a)
      end
    end
  end

  context 'with `last_batch: :rollover`' do
    let(:last_batch) do
      :rollover
    end

    before do
      loader.to_a
    end

    describe '#length' do
      it 'is 33 (mini-batches)' do
        expect(loader.length).to eq(33)
      end
    end

    describe '#each' do
      it 'results in an array of slices' do
        expect(loader.each.map(&:to_a)).to eq((0..99).to_a.rotate(-1).each_slice(3).to_a[0..-2])
      end
    end
  end
end
