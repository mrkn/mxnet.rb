require 'json'

module MXNet
  module Registry
    @registries = {}

    class Manager
      def initialize(base_class, nickname)
        @registry = registry_for(base_class)
        @base_class = base_class
        @nickname = nickname
      end

      private def registry_for(klass)
        registries = MXNet::Registry.instance_variable_get(:@registries)
        registries[klass] ||= {}
      end

      # Register functions
      def register(klass, name=nil)
        unless klass < @base_class
          raise TypeError, "Can only register subclass of #{@base_class}"
        end
        name = klass.name[/:?(\w+)\z/, 1] if name.nil?
        name = name.downcase.to_sym
        if @registry.has_key?(name)
          warn "\033[91mNew #{@nickname} #{klass} registered with name #{name}" +
               " is overriding existing #{@nickname} #{@registry[name]}\033[0m"
        end
        @registry[name] = klass
      end

      def alias(klass, *aliases)
        aliases.each do |name|
          register(klass, name)
        end
        klass
      end

      def create(*args, **kwargs)
        if args.length > 0
          name = args.shift
        else
          name = kwargs.delete(@nickname)
        end

        case name
        when @base_class
          unless args.length == 0  && kwargs.length == 0
            raise ArgumentError,
                  "#{@nickname} is already an instance. " +
                  "Additional arguments are invalid"
          end
          return name
        when Hash
          return create(**name)
        when String, ::Symbol
          name = name.to_s
        else
          if name.respond_to?(:to_str)
            name = name.to_str
          else
            raise ArgumentError, "#{@nickname} must be a String or a Symbol (#{name.inspect})"
          end
        end

        # The variable `name`` is ensured to be a String, here.

        if name.start_with?('[') || name.start_with?('{')
          unless args.length == 0 && kwargs.length == 0
            raise ArgumentError, "Additional arguments for JSON is invalid."
          end
          kwargs = JSON.load(name)
          if kwargs.is_a?(Hash)
            return create(**symbolize_hash_keys(kwargs))
          else
            name, kwargs = *kwargs
            kwargs = symbolize_hash_keys(kwargs || {})
            return create(name, **kwargs)
          end
        end

        name = name.downcase.to_sym
        unless @registry.has_key?(name)
          raise ArgumentError,
                "#{name} is not registered. " +
                "Please register it for #{@nickname} first"
        end
        return @registry[name].new(*args, **kwargs)
      end

      private

      if Hash.instance_methods.include? :transform_keys
        def symbolize_hash_keys(hash)
          hash.transform_keys(&:to_sym)
        end
      else
        def symbolize_hash_keys(hash)
          {}.tap do |result|
            hash.each do |key, value|
              result[key.to_sym] = value
            end
          end
        end
      end
    end
  end
end
