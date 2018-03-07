module MXNet
  module IO
    # A ruby wrapper of a C++ data iterator.
    class MXDataIter < DataIter
      def initialize(data_name: 'data', label_name: 'softmax_label', **)
        @debug_skip_load = false

        # load the first batch to get shape information
        @first_batch = next_batch
        data = @first_batch.data[0]
        label = @first_batch.label[0]

        # properties
        @provide_data = [DataDesc.new(data_name, data.shape, dtype: data.dtype)]
        @provide_label = [DataDesc.new(label_name, label.shape, dtype: label.dtype)]

        super(batch_size: data.shape[0])
      end

      def debug_skip_load
        # Set the iterator to simply return always first batch.  This can be used
        # to test the speed of network without taking the loading delay into
        # account.
        @debug_skip_load = true
        # logging.info("Set debug_skip_load to be true, will simply return first batch")
      end

      def reset
        @debug_at_begin = true
        @first_batch = nil
        _reset
      end

      def next_batch
        if @debug_skip_load && !@debug_at_begin
          return DataBatch.new([current_data], label: [current_label], pad: current_pad, index: current_index)
        end
        if @first_batch
          batch = @first_batch
          @first_batch = nil
          return batch
        end
        @debug_at_begin = false
        if _iter_next > 0
          DataBatch.new([current_data], label: [current_label], pad: current_pad, index: current_index)
        end
      end

      def iter_next
        return true if @first_batch
        _iter_next
      end

      def current_data
        _current_data
      end

      def current_label
        _current_label
      end

      def current_pad
        _current_label
      end
    end
  end
end
