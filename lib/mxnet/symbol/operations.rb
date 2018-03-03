require 'forwardable'

module MXNet
  class Symbol
    class << self
      extend Forwardable

      def_delegators :"MXNet::Symbol::Ops", *Ops.methods(false)
    end
  end
end
