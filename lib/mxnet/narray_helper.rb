require 'numo/narray'
require 'mxnet/narray_helper.so'

module MXNet
  module NArrayHelper
    module_function

    MXNET_DTYPE_TO_NUMO = {
      float32: Numo::SFloat,
      float64: Numo::DFloat,
      float16: nil,  # TODO
      uint8: Numo::UInt8,
      int32: Numo::Int32,
      int8: Numo::Int8,
      int64: Numo::Int64,
    }.freeze

    def to_ndarray(nary, ctx:, dtype:)
      nary_type = MXNET_DTYPE_TO_NUMO[dtype]
      if nary_type.nil?
        raise ArgumentError,
          "converting #{nary.class} to MXNet::NDArray(dtype: #{dtype}) is unsupported"
      end

      nd = MXNet::NDArray.empty(nary.shape, ctx: ctx, dtype: dtype)
      nary = nary_type.cast(nary) unless nary.class <= nary_type
      sync_copyfrom(nd, nary)
    end

    # TODO: move to NumBuffer
    NDArray::CONVERTER << [Numo::NArray, NArrayHelper]
  end
end
