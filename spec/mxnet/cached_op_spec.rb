require 'spec_helper'

RSpec.describe MXNet::CachedOp do
  describe '#new' do
    let(:sym) do
      MXNet::Symbol.var('sym')
    end
    it 'accepts keyword arguments' do
      expect{MXNet::CachedOp.new(sym, static_alloc: 1)}.not_to raise_error
      expect{MXNet::CachedOp.new(sym, data_indices: [0])}.not_to raise_error
    end
  end
  describe '#call' do
    let(:sym) do
      MXNet::Symbol.var('sym')
    end
    let(:sqr) do
      MXNet::CachedOp.new(sym * sym)
    end
    it 'squares the input' do
      input = MXNet::NDArray.array([1, 2, 3])
      expect(sqr.call(input).to_a).to eq([1, 4, 9])
    end
    it 'expects the correct number of inputs' do
      inputs = [MXNet::NDArray.array([4, 5, 6]), MXNet::NDArray.array([7, 8, 9])]
      expect{sqr.call(*inputs)}.to raise_error(MXNet::Error, /\(2 vs\. 1\)/)
      expect{sqr.call()}.to raise_error(MXNet::Error, /\(0 vs\. 1\)/)
    end
  end
end
