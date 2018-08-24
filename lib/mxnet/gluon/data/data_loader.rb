require_relative 'sampler'

module MXNet
  module Gluon
    module Data
      # Loads data from a dataset and returns mini-batches of data.
      class DataLoader
        include Enumerable

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
        def initialize(dataset, batch_size: nil, shuffle: false, sampler: nil,
                       last_batch: nil, batch_sampler: nil, batchify_fn: nil,
                       num_workers: 0)
          @dataset = dataset

          if batch_sampler.nil?
            unless batch_size
              raise ArgumentError, "batch_size must be specified unless " +
                    "batch_sampler is specified"
            end
            if sampler.nil?
              if shuffle
                sampler = MXNet::Gluon::Data::RandomSampler.new(dataset.length)
              else
                sampler = MXNet::Gluon::Data::SequentialSampler.new(dataset.length)
              end
            elsif shuffle
              raise ArgumentError, "shuffle must not be specified if " +
                    "sampler is specified"
            end

            batch_sampler = MXNet::Gluon::Data::BatchSampler.new(
              sampler, batch_size, last_batch || :keep)
          elsif batch_size || shuffle || sampler || last_batch
            raise ArgumentError,
                  "batch_size, shuffle, sampler, and last_batch must not be " +
                  "specified if batch_sampler is specified."
          end

          @batch_sampler = batch_sampler
          @num_workers = [num_workers, 0].max
          if batchify_fn.nil?
            if num_workers > 0
              @batchify_fn = method(:default_mp_batchify_fn)
            else
              @batchify_fn = method(:default_batchify_fn)
            end
          else
            @batchify_fn = batchify_fn
          end
        end

        def each
          return enum_for unless block_given?
          raise NotImplementedError
        end

        def length
          @batch_sampler.length
        end
      end
    end
  end
end
