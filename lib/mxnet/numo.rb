require 'numo/narray'
require 'mxnet/ndarray'

module MXNet
  class NDArray
    DTYPE_NUMO_TO_MXNET = {
      Numo::SFloat => DType::FLOAT32,
      Numo::DFloat => DType::FLOAT64,
      # numo-narray doesn't have half-float type
      Numo::UInt8  => DType::UINT8,
      Numo::Int32  => DType::INT32,
      Numo::Int8   => DType::INT8,
      Numo::Int64  => DType::INT64
    }.freeze

    DTYPE_MXNET_TO_NUMO = {
      DType::FLOAT32 => Numo::SFloat,
      DType::FLOAT64 => Numo::DFloat,
      DType::FLOAT16 => Numo::SFloat, # numo-narray doesn't have half-float type
      DType::UINT8 => Numo::UInt8,
      DType::INT32 => Numo::Int32,
      DType::INT8 => Numo::Int8,
      DType::INT64 => Numo::Int64
    }.freeze
  end
end
