module MXNet
  module IO
    # DataDesc is used to store name, shape, type and layout information
    # of the data or the label.
    class DataDesc < Struct.new(:name, :shape)
      def initialize(name, shape, dtype: :float32, layout: 'NCHW')
        super(name, shape)
        @dtype = dtype
        @layout = layout
      end

      attr_reader :dtype, :layout

      def inspect
        "#{self.class}[#{name}, #{shape}, #{dtype}, #{layout}]"
      end
    end

    # A data batch
    class DataBatch
      def initialize(data, label: nil, pad: nil, index: nil, bucket_key: nil, provide_data: nil, provide_label: nil)
        check_data(data)
        check_label(label)
        @data = data
        @label = label
        @pad = pad
        @index = index

        @bucket_key = bucket_key
        @provide_data = provide_data
        @provide_label = provide_label
      end

      attr_reader :data, :label, :pad, :index

      private def check_data(data)
        return unless data
        return if data.is_a? Array
        raise TypeError, "Data must be a list of NDArrays"
      end

      private def check_label(label)
        return unless label
        return if label.is_a? Array
        raise TypeError, "Label must be a list of NDArrays"
      end

      def inspect
        super.tap do |s|
	  data_shapes = @data&.map {|d| d.shape } || []
	  label_shapes = @label&.map {|l| l.shape } || []
          s[-1, 0] = "data shapes: #{data_shapes} label_shapes: #{label_shapes}"
        end
      end
    end

    # The base class for an MXNet data iterator.
    class DataIter
      include Enumerable

      # @param batch_size: [Integer] The batch size, namely the number of items in the batch.
      def initialize(batch_size: 0)
        @batch_size = batch_size
      end

      attr_reader :batch_size

      def each
        return enum_for unless block_given?

        while batch = self.next_batch
          yield batch
        end
      ensure
        reset
      end

      # Get next data batch from iterator.
      #
      # @return [DataBatch, nil] The data of next batch.
      #   If the end of the data is reached, return `nil`.
      def next_batch
        if iter_next
          DataBatch.new(current_data, label: current_label, pad: current_pad, index: current_index)
        end
      end

      # Move to the next batch.
      #
      # @return [true, false] Whether the move is successful.
      def iter_next
        false
      end

      # Reset the iterator to the begin of the data.
      def reset
        nil
      end
      alias_method :rewind, :reset

      # Get data of the current batch.
      #
      # @return [Array<MXNet::NDArray>] The data of the current batch.
      def current_data
        nil
      end

      # Get label of the current batch.
      #
      # @return [Array<MXNet::NDArray>] The label of the current batch.
      def current_label
        nil
      end

      # Get index of the current batch.
      #
      # @return [Array<Integer>] The indices of examples in the current batch.
      def current_index
        nil
      end

      # Get the number of padding examples in the current batch.
      #
      # @return [Integer] Number of padding examples in the current batch.
      def current_pad
        nil
      end
    end
  end
end

# require 'mxnet/io/resize_iter'
# require 'mxnet/io/prefetching_iter'
require 'mxnet/io/ndarray_iter'
require 'mxnet/io/mxdata_iter'
