module MXNet
  module IO
    # Resize a data iterator to a given number of batches.
    class ResizeIter < DataIter
      def initialize(data_iter, size, reset_internal: true)
        super()
        @data_iter = data_iter
        @size = size
        @reset_internal = reset_internal
        @cur = 0
        @current_batch = nil

        @provide_data = data_iter.provide_data?
        @provide_label = data_iter.provide_label?
        @batch_size = data_iter.batch_size
        if data_iter.respond_to? :default_bucket_key
          @default_bucket_key = data_iter.default_bucket_key
        end
      end

      attr_reader :provide_data, :provide_label, :default_bucket_key
      alias_method :provide_data?, :provide_data
      alias_method :provide_label?, :provide_label

      def reset
        @cur = 0
        @data_iter.reset if @reset_internal
      end

      def iter_next
        return false if @cur == @size
        if (batch = @data_iter.next)
          @current_batch = batch
        else
          @data_iter.reset
          @current_batch = @data_iter.next
        end
        @cur += 1
        true
      end

      def current_data
        @current_batch.data
      end

      def current_label
        @current_batch.label
      end

      def current_index
        @current_batch.index
      end

      def current_pad
        @current_batch.pad
      end
    end
  end
end
