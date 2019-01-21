module MXNet
  module Name
    # NameManager to do automatic naming.
    #
    # Developers can also inherit from this class to change naming behavior.
    class NameManager
      @current = nil

      class << self
        attr_accessor :current
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
        # TODO: use next_count_for
        @counter[hint] = 0 unless @counter.include? hint
        name = "#{hint}#{@counter[hint]}"
        @counter[hint] += 1
        return name
      end

      def next_count_for(hint)
        @counter[hint] || 0
      end

      def enter
        return _enter unless block_given?
        begin
          _enter
          yield
        ensure
          exit
        end
      end

      private def _enter
        @old_manager = NameManager.current
        NameManager.current = self
        self
      end

      def exit
        raise unless @old_manager
        NameManager.current = @old_manager
      end

      @current = self.new
    end

    # A name manager that attaches a prefix to all names.
    class Prefix < NameManager
      def initialize(prefix:)
        super()
        @prefix = prefix
      end

      def get(name, hint)
        name = super(name, hint)
        "#{@prefix}#{name}"
      end
    end
  end
end
