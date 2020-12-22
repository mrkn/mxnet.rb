module MXNet
  module IO
    # Returns an iterator for mx.nd.NDArray.
    class NDArrayIter < DataIter
      attr_reader :data
      def initialize(data, label: nil, batch_size: 1, shuffle: false, last_batch_handle: 'pad', data_name: 'data', label_name: 'softmax_label')
        # data and label are NDArray
        # last_batch_handle = 'pad' or 'discard'

        @data = data
        @label = label
        @shuffle = shuffle
        
        @num_data = @data.shape[0]
        
        # properties
        @provide_data = [DataDesc.new(data_name, @data[0].shape, dtype: @data[0].dtype)]
        @provide_label = [DataDesc.new(label_name, @label[0].shape, dtype: @label[0].dtype)] if @label

        super(batch_size: batch_size)

        reset()
      end

      def reset
        @cursor = -@batch_size
        if @shuffle
          seed = (rand() * 100000).to_i
          @data.shuffle(Random.new(seed))
          @label.shuffle(Random.new(seed))
        end
      end

      def iter_next
        @cursor += @batch_size
        return @cursor < @num_data
      end

      def current_data
        return self._batchify(@data)
      end

      def current_label
        return self._batchify(@label)
      end

      def current_pad
        if @last_batch_handle == 'pad' and @cursor + @batch_size > @num_data
          return @cursor + @batch_size - @num_data
        else
          return 0
        end
      end
    private
      def _batchify(data_source)
        """Load data from underlying arrays, internal use only."""
        raise 'DataIter needs reset.' unless @cursor < @num_data
        # last batch with 'pad'
        if @last_batch_handle == 'pad' and \
          @cursor + @batch_size > @num_data
          pad = @batch_size - @num_data + @cursor
          return MXNet::NDArray.concat(data_source.slice(begin: @cursor, end: @num_data), data_source.slice(begin:0, end: pad), dim: 0)
        # normal case
        else
          if @cursor + @batch_size < @num_data
            slice_length = @batch_size
          else
            # get incomplete last batch
            slice_length = @num_data - @cursor
          end
          return data_source.slice(begin: @cursor, end: slice_length + @cursor)
        end
      end
    end
  end
end
