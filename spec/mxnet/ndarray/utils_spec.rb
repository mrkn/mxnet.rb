require 'spec_helper'

RSpec.describe MXNet::NDArray do
  describe '.save and .load', :within_tmpdir do
    context 'when the data is a NDArray' do
      specify do
        data = MXNet::NDArray.array([[1, 2], [3, 4]])
        MXNet::NDArray.save('ndarray', data)
        expect(File.file?('ndarray')).to eq(true)
        loaded = MXNet::NDArray.load('ndarray')
        expect(loaded).to be_an(Array)
        expect(loaded.length).to eq(1)
        expect(loaded[0]).to eq(data)
      end
    end

    context 'when the data is an array of NDArrays' do
      specify do
        data = [
          MXNet::NDArray.array([[1, 2], [3, 4]]),
          MXNet::NDArray.array([[1, 2, 3], [4, 5, 6]])
        ]
        MXNet::NDArray.save('list', data)
        expect(File.file?('list')).to eq(true)
        loaded = MXNet::NDArray.load('list')
        expect(loaded).to be_an(Array)
        expect(loaded.length).to eq(2)
        expect(loaded).to eq(data)
      end
    end

    context 'when the data is a hash of String => NDArray' do
      specify do
        data = {
          x: MXNet::NDArray.array([[1, 2], [3, 4]]),
          y: MXNet::NDArray.array([[1, 2, 3], [4, 5, 6]])
        }
        MXNet::NDArray.save('dict', data)
        expect(File.file?('dict')).to eq(true)
        loaded = MXNet::NDArray.load('dict')
        expect(loaded).to be_an(Hash)
        expect(loaded.length).to eq(2)
        expect(loaded).to eq(stringify_hash_keys(data))
      end
    end
  end

  def stringify_hash_keys(hash)
    MXNet::Utils.transform_hash_keys(hash, &:to_s)
  end
end
