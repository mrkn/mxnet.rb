require 'forwardable'

module MXNet
  class NDArray
    class << self
      extend Forwardable

      def_delegators :"MXNet::NDArray::Ops", *Ops.methods(false)
    end
  end
end
