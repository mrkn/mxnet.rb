require 'mxnet/gluon/data'
require 'mxnet/narray_helper'

require 'zlib'
require 'numo/narray'

module MXNet::Gluon::Data
  module Vision
    class MNIST < DownloadedDataset
      def initialize(root: File.join('~', '.mxnet', 'datasets', 'mnist'),
                     train: true, transform: nil)
        @train = train
        @train_data = ['train-images-idx3-ubyte.gz',
                       '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d']
        @train_label = ['train-labels-idx1-ubyte.gz',
                        '2a80914081dc54586dbdf242f9805a6b8d2a15fc']
        @test_data = ['t10k-images-idx3-ubyte.gz',
                      'c3a25af1f52dad7f726cce8cacb138654b760d48']
        @test_label = ['t10k-labels-idx1-ubyte.gz',
                       '763e7fa3757d93b0cdec073cef058b2004252c17']
        @namespace = 'mnist'
        super(root: root, transform: transform)
      end

      attr_reader :train

      private def _get_data
        if @train
          data, label = @train_data, @train_label
        else
          data, label = @test_data, @test_label
        end

        namespace = "gluon/dataset/#{@namespace}"
        data_file = MXNet::Utils.download(
          MXNet::Utils.get_repo_file_url(namespace, data[0]),
          path: self.root,
          sha1_hash: data[1]
        )
        label_file = MXNet::Utils.download(
          MXNet::Utils.get_repo_file_url(namespace, label[0]),
          path: self.root,
          sha1_hash: label[1]
        )

        Zlib::GzipReader.open(label_file) do |label_reader|
          magic = read_magic(label_reader)
          count = read_count(label_reader)
          @label = Numo::UInt8.from_binary(label_reader.read)
        end

        Zlib::GzipReader.open(data_file) do |data_reader|
          magic = read_magic(data_reader)
          count = read_count(data_reader)
          nrows, ncols = read_image_size(data_reader)
          data = Numo::UInt8.from_binary(data_reader.read, [@label.length, nrows, ncols, 1])
          @data = MXNet::NDArray(data)
        end
      end

      private def read_uint32(reader)
        reader.read(4).unpack('N*')
      end

      alias_method :read_magic, :read_uint32
      alias_method :read_count, :read_uint32

      private def read_image_size(reader)
        reader.read(8).unpack('N*')
      end
    end
  end
end
