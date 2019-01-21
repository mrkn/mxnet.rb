module MXNet
  module Gluon
    module Data
      # Base class for samplers.
      #
      # All samplers should subclass Sampler and define #length and
      # #each methods.
      #
      class Sampler
        include Enumerable

        def length
          raise NotImplementedError
        end

        def each
          raise NotImplementedError
        end
      end

      # Samples elements from [0, length) sequentially.
      class SequentialSampler < Sampler
        # Creates a new instance.
        #
        # ====Parameters
        #
        # +length+:: (integer)
        #            Length of the sequence.
        #
        def initialize(length)
          @length = length
        end

        def length
          @length
        end

        def each(&block)
          return enum_for unless block_given?
          @length.times.each(&block)
        end
      end

      # Samples elements from [0, length) randomly without
      # replacement.
      #
      class RandomSampler < Sampler
        # Creates a new instance.
        #
        # ====Parameters
        #
        # +length+:: (integer)
        #            Length of the sequence.
        #
        def initialize(length)
          @length = length
        end

        def length
          @length
        end

        def each(&block)
          return enum_for unless block_given?
          indices = @length.times.to_a
          indices.shuffle!
          indices.each(&block)
        end
      end

      # Wraps another Sampler and return mini-batches of samples.
      #
      #     sampler = MXNet::Gluon::Data.SequentialSampler(10)
      #     batch_sampler = MXNet::Gluon::Data.BatchSampler(sampler, 3, :keep)
      #     batch_sampler.to_a # => [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
      #
      class BatchSampler < Sampler
        # Creates a new instance.
        #
        # ====Parameters
        #
        # +sampler+::    (Sampler)
        #                The source Sampler.
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
        def initialize(sampler, batch_size, last_batch: :keep)
          unless [:keep, :discard, :rollover].include?(last_batch)
            raise ArgumentError, 'last_batch must be either :keep, :discard, or :rollover'
          end
          @sampler = sampler
          @batch_size = batch_size
          @last_batch = last_batch
          @prev = []
        end

        def length
          case @last_batch
          when :discard
            @sampler.length.div(@batch_size)
          when :keep
            (@sampler.length + @batch_size - 1).div(@batch_size)
          when :rollover
            (@sampler.length + @prev.length).div(@batch_size)
          end
        end

        def each
          return enum_for unless block_given?

          batch, @prev = @prev, []
          @sampler.each do |i|
            batch << i
            if batch.length == @batch_size
              yield batch
              batch = []
            end
          end

          unless batch.empty?
            case @last_batch
            when :discard
            when :keep
              yield batch
            when :rollover
              @prev = batch
            end
          end
        end
      end
    end
  end
end
