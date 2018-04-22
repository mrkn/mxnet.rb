require 'fileutils'

module Utils
  module_function def get_cifar10(data_dir)
    unless File.directory? data_dir
      FileUtils.mkdir_p data_dir
    end
    Dir.chdir data_dir do
      unless File.exist?('train.rec') && File.exist?('test.rec') &&
             File.exist?('train.lst') && File.exist?('test.lst')

        # Download CIFAR10 data
        MXNet::Utils.download(
          'http://data.mxnet.io/mxnet/data/cifar10.zip',
          sha1_hash: 'b9ac287012f2dad9dfb49d8271c39ecdd7db376c')
        system 'unzip', '-x', 'cifar10.zip'
        Dir.glob('cifar/*') do |fn|
          system 'mv', fn, File.basename(fn)
        end
        FileUtils.rmdir 'cifar'
      end
    end
  end

  module_function def get_cifar10_iter(batch_size:, data_dir:, num_workers: 1, rank: 0)
    train = MXNet::IO::ImageRecordIter.new(
      path_imgrec: File.join(data_dir, 'train.rec'),
      path_imglst: File.join(data_dir, 'train.lst'),
      data_shape: [3, 32, 32],
      batch_size: batch_size,
      rand_crop: true,
      rand_mirror: true,
      num_parts: num_workers,
      part_index: rank
    )
    val = MXNet::IO::ImageRecordIter.new(
      path_imgrec: File.join(data_dir, 'test.rec'),
      path_imglst: File.join(data_dir, 'test.lst'),
      data_shape: [3, 32, 32],
      batch_size: batch_size,
      rand_crop: false,
      rand_mirror: false,
      num_parts: num_workers,
      part_index: rank
    )
    return [train, val]
  end
end
