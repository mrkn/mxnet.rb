require 'mxnet/gluon/data'
require 'mxnet/gluon/data/sampler'

module MXNet
  module Gluon
    module Data
      ##
      # Loads data from a Dataset and returns mini-batches of data.
      #
      class DataLoader
        include Enumerable

        ##
        # Creates a new instance.
        #
        # ====Parameters
        #
        # +dataset+::    (Dataset, NDArray or array)
        #                Source. Note that instances of NDArray and
        #                Array can be used directly.
        # +shuffle+::    (boolean)
        #                Whether or not to shuffle the samples.
        # +batch_size+:: (integer)
        #                Size of mini-batch.
        # +last_batch+:: (+:keep+, +:discard+, +:rollover+)
        #                Specifies how the last batch is handled if
        #                +batch_size+ does not evenly divide sequence
        #                length. If +:keep+, the last batch will be
        #                returned directly, but will contain fewer
        #                elements than +batch_size+ requires. If
        #                +:discard+, the last batch will be discarded.
        #                If +:rollover+, the remaining elements will
        #                be rolled over to the next iteration.
        #
        def initialize(dataset, shuffle:, batch_size:, last_batch: :keep)
          @dataset = dataset
          @sampler = shuffle ?
                       MXNet::Gluon::Data.RandomSampler(dataset.length) :
                       MXNet::Gluon::Data.SequentialSampler(dataset.length)
          @sampler = MXNet::Gluon::Data.BatchSampler(@sampler, batch_size, last_batch)
        end

        def length
          @sampler.length
        end

        def each
          return enum unless block_given?
          enum.each do |i|
            yield i
          end
        end

        private

        def enum
          Enumerator.new do |yielder|
            @sampler.each do |batch|
              yielder << batchify(batch.map { |i| @dataset[i] })
            end
          end
        end

        def batchify(data)
          case data[0]
          when Array
            data[0].zip(*data[1..-1]).map { |d| batchify(d) }
          when MXNet::NDArray
            MXNet::NDArray.stack(*data)
          else
            MXNet::NDArray.array(data.to_a)
          end
        end
      end

      def self.DataLoader(*args)
        DataLoader.new(*args)
      end
    end
  end
end
