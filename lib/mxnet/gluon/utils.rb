require 'mxnet/gluon'

module MXNet
  module Gluon
    module Utils
      # Splits an NDArray into `num_slice` slices along `batch_axis`.
      # Usually used for data parallelism where each slices is sent
      # to one device (i.e. GPU).
      #
      # Parameters
      # ----------
      # data : NDArray
      #     A batch of data.
      # num_slice : int
      #     Number of desired slices.
      # batch_axis : int, default 0
      #     The axis along which to slice.
      # even_split : bool, default True
      #     Whether to force all slices to have the same number of elements.
      #     If `True`, an error will be raised when `num_slice` does not evenly
      #     divide `data.shape[batch_axis]`.
      #
      # Returns
      # -------
      # list of NDArray
      #     Return value is a list even if `num_slice` is 1.
      module_function def split_data(data, num_slice, batch_axis=0, even_split: true)
        size = data.shape[batch_axis]
        if size < num_slice
          raise ArgumentError,
                "Too many slices for data with shape #{data.shape}. " +
                "Arguments are num_slice=#{num_slice} and " +
                "batch_axis=#{batch_axis}."
        elsif even_split && size % num_slice != 0
          raise ArgumentError,
                "data with shape #{data.shape} cannot be evenly split into " +
                "#{num_slice} slices along axis #{batch_axis}. " +
                "Use a batch size that's multiple of #{num_slice} or set " +
                "even_split: false to allow uneven partitioning of data."
        end

        step = size.div(num_slice)
        if batch_axis == 0
          slices = (0...num_slice).map do |i|
            b = i * step
            e = (i < num_slice - 1) ? (i+1) * step : size
            data[b ... e]
          end
        elsif even_split
          slices = MXNet::NDArray.split(data, num_outputs: num_slice, axis: batch_axis)
        else
          slices = (0...num_slice).map do |i|
            b = i * step
            e = (i < num_slice - 1) ? (i+1) * step : size
            MXNet::NDArray.slice_axis(data, axis: batch_axis, begin: b, end: e)
          end
        end
        return slices
      end
    end
  end
end
