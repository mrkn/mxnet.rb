module MXNet
  module Gluon
    module Data
      # Loads data from a dataset and returns mini-batches of data.
      class DataLoader
        include Enumerable

        def initialize(dataset, batch_size: nil, shuffle: nil, sampler: nil,
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
                sampler = _sampler.RandomSampler.new(dataset.length)
              else
                sampler = _sampler.SequentialSampler.new(dataset.length)
              end
            elsif shuffle
              raise ArgumentError, "shuffle must not be specified if " +
                    "sampler is specified"
            end

            batch_sampler = _sampler.BatchSampler(
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
