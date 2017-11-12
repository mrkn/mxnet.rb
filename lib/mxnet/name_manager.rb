module MXNet
  # NameManager to do automatic naming.
  #
  # Developers can also inherit from this class to change naming behavior.
  class NameManager
    @current = nil

    class << self
      attr_reader :current
    end

    def initialize
      @counter = {}
      @old_manager = nil
    end

    # Get the canonical name for a symbol.
    #
    # This is the default implementation.
    # If the user specifies a name,
    # the user-specified name will be used.
    #
    # When user does not specify a name, we automatically generate a
    # name based on the hint string.
    #
    # @param name [String|nil]  The name specified by the user.
    # @param hint [String]  A hint string, which can be used to generate name.
    #
    # @return [String]  A canonical name for the symbol.
    def get(name, hint)
      return name if name
      @counter[hint] = 0 unless @counter.include? hint
      name = "#{hint}#{@counter[hint]}"
      @counter[hint] += 1
      return name
    end

    @current = self.new
  end
end
