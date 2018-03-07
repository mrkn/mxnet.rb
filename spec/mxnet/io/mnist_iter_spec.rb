require 'spec_helper'

require 'fileutils'
require 'open-uri'
require 'pathname'
require 'zlib'

::RSpec.describe MXNet::IO::MNISTIter do
  def download(url, dirname: Dir.pwd)
    url = String(url)
    FileUtils.mkdir_p(dirname)
    basename = File.basename(url)
    fullname = File.join(dirname, basename)
    URI.parse(url).open('rb') do |sio|
      open(fullname, 'wb') do |io|
        while chunk = sio.read(1024)
          io.write(chunk)
        end
      end
    end
  end

  let(:fixture_dir_path) do
    Pathname(File.expand_path('../../../fixture' , __FILE__))
  end

  before do
    mnist_dir = fixture_dir_path.join('mnist')
    mnist_dir.mkdir unless mnist_dir.exist?
    Dir.chdir mnist_dir do
      unless File.exist?('train-images-idx3-ubyte') &&
             File.exist?('train-labels-idx1-ubyte')
        # Download MNIST data
        download('http://data.mxnet.io/mxnet/data/mnist.zip')
        # Extract MNIST data
        system('unzip -x mnist.zip')
      end
    end
  end

  let(:batch_size) { 100 }

  subject(:train_iter) do
    MXNet::IO::MNISTIter.new(
      image: fixture_dir_path.join('mnist/train-images-idx3-ubyte'),
      label: fixture_dir_path.join('mnist/train-labels-idx1-ubyte'),
      data_shape: [784],
      batch_size: batch_size,
      shuffle: 1,
      flat: 1,
      silent: 0,
      seed: 10
    )
  end

  specify 'batch size' do
    nbatch = 60000 / batch_size
    batch_count = 0
    train_iter.each { batch_count += 1 }
    expect(batch_count).to eq(nbatch)
  end

  specify do
    train_iter.iter_next
    label_0 = train_iter.current_label.to_narray
    4.times { train_iter.iter_next }
    train_iter.reset
    train_iter.iter_next
    label_1 = train_iter.current_label.to_narray
    expect((label_0 - label_1).sum).to eq(0)
  end
end
