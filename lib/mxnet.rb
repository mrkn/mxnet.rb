require "mxnet/version"

module MXNet
  module HandleWrapper
  end

  None = Object.new
  def None.to_s
    'None'
  end
  def None.inspect
    'None'
  end

  require 'mxnet/libmxnet'
  require 'mxnet/attribute'
  require 'mxnet/autograd'
  require 'mxnet/context'
  require 'mxnet/name/name_manager'
  require 'mxnet/executor'
  require 'mxnet/initializer'
  require 'mxnet/io'
  require 'mxnet/metric'
  require 'mxnet/ndarray'
  require 'mxnet/ndarray/operation_delegator'
  require 'mxnet/optimizer'
  require 'mxnet/symbol'
  require 'mxnet/symbol/operation_delegator'
  require 'mxnet/random'
  require 'mxnet/utils'
  require 'mxnet/op_info'
  require 'mxnet.so'
  require 'mxnet/ndarray/operations'
  require 'mxnet/symbol/operations'
  require 'mxnet/lr_scheduler'
end
