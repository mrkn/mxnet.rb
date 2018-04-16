require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Data::Vision::MNIST do
  context 'the default settings' do
    subject(:dataset) do
      MXNet::Gluon::Data::Vision::MNIST.new
    end

    specify do
      expect(dataset.root).to eq(File.expand_path("~/.mxnet/datasets/mnist"))
      expect(dataset.train).to eq(true)
    end
  end

  subject(:dataset) do
    MXNet::Gluon::Data::Vision::MNIST.new(train: train, transform: transform)
  end

  let(:train) { true }
  let(:transform) { nil }

  describe '#length' do
    specify do
      expect(dataset.length).to eq(60000)
    end

    context 'with train: false' do
      let(:train) { false }

      specify do
        expect(dataset.length).to eq(10000)
      end
    end
  end

  describe '#[]' do
    specify do
      record = dataset[0]
      expect(record).to be_an(Array)
      expect(record.length).to eq(2)
      expect(record[0]).to be_a(MXNet::NDArray)
      expect(record[0].shape).to eq([28, 28, 1])
      expect(record[1]).to eq(5)
      expect(dataset[59999][1]).to eq(8)
    end

    context 'with train: false' do
      let(:train) { false }

      specify do
        record = dataset[0]
        expect(record).to be_an(Array)
        expect(record.length).to eq(2)
        expect(record[0]).to be_a(MXNet::NDArray)
        expect(record[0].shape).to eq([28, 28, 1])
        expect(record[1]).to eq(7)
        expect(dataset[9999][1]).to eq(6)
      end
    end
  end
end
