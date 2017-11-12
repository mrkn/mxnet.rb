module MXNet
  class AttrScope
    @current = nil

    def self.current
      @current
    end

    def initialize(**kwargs)
      @old_scope = nil
      kwargs.each_value do |v|
        raise ArgumentError, "attributes need to be string" unless v.kind_of?(String) or v.kind_of?(Symbol)
      end
      @attr = kwargs
    end

    # Get the hash of attributes given the attribute set by the symbol.
    #
    # @param attr [Hash]  The attribute passed in by user during symbol creation.
    # @return [Hash]  Updated attributes to add other scope related attributes.
    def get(attr)
      if @attr
        res = @attr.dup
        attr ? res.update(attr) : res
      else
        attr || {}
      end
    end

    @current = self.new
  end
end
