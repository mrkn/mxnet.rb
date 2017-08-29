require 'numo/narray'
require 'mxnet/ndarray'

module MXNet
  class NDArray
    DTYPE_NUMO_TO_MXNET = {
      Numo::SFloat => :float32,
      Numo::DFloat => :float64
      # numo-narray doesn't have half-float type
      Numo::UInt8  => :uint8
      Numo::Int32  => :int32
      Numo::Int8   => :int8
      Numo::Int64  => :int64
    }.freeze

    DTYPE_MXNET_TO_NUMO = {
      float32: Numo::SFloat,
      float64: Numo::DFloat,
      float16: Numo::SFloat, # numo-narray doesn't have half-float type
      uint8: Numo::UInt8,
      int32: Numo::Int32,
      int8: Numo::Int8,
      int64: Numo::Int64
    }.freeze
  end
end
