require 'spec_helper'

RSpec.describe MXNet::IO::ImageRecordIter do
  # TODO: should be refactord with the same things in mnist_iter_spec.rb
  let(:fixture_dir_path) do
    Pathname(File.expand_path('../../../fixture' , __FILE__))
  end

  before do
    cifar10_dir = fixture_dir_path.join('cifar10')
    cifar10_dir.mkdir unless cifar10_dir.exist?
    Dir.chdir cifar10_dir do
      unless File.exist?('train.rec') && File.exist?('test.rec') &&
             File.exist?('train.lst') && File.exist?('test.lst')

        # Download CIFAR10 data
        MXNet::Utils.download(
          'http://data.mxnet.io/mxnet/data/cifar10.zip',
          sha1_hash: 'b9ac287012f2dad9dfb49d8271c39ecdd7db376c'
        )
        system 'unzip', '-x', 'cifar10.zip', out: File::NULL
        Dir.glob('cifar/*') do |fn|
          system 'mv', fn, File.basename(fn)
        end
      end
    end
  end

  context 'for CIFAR10 train images' do
    let(:batch_size) { 100 }

    subject(:cifar10_iter) do
      data_iter = MXNet::IO::ImageRecordIter.new(
        path_imgrec: fixture_dir_path.join('cifar10/train.rec'),
        path_imglst: fixture_dir_path.join('cifar10/train.lst'),
        data_shape: [3, 32, 32],
        batch_size: batch_size,
        rand_crop: true,
        rand_mirror: true,
        num_parts: 1,
        part_index: 0
      )
    end

    specify 'batch count' do
      batch_count = 0
      cifar10_iter.each { batch_count += 1 }
      expect(batch_count).to eq(50000 / batch_size)
    end

    specify do
      cifar10_iter.iter_next
      batch_data = cifar10_iter.current_data
      expect(batch_data.shape).to eq([batch_size, 3, 32, 32])
      batch_label = cifar10_iter.current_label
      expect(batch_label.shape).to eq([batch_size])
      expect(batch_label[0].as_scalar).to eq(2)
    end
  end
end
