require 'mxnet'

module MXNet
  ##
  # The base class of an initializer.
  #
  # Custom initializers can be created by subclassing Initializer and
  # implementing the required function #init_array. By default, the
  # created initializer will be registered under its simplified class
  # name (`class.name.split('::').last.downcase.to_sym`) but it may be
  # registered under another name by calling #register.
  #
  #     class CustomInit < Initializer
  #       register :myinit
  #       def init_array(array)
  #         array[0..-1] = 1.0
  #       end
  #     end
  #
  class Initializer
    def self.inherited(child)
      default = child.name.split('::').last.downcase
      child.register(default)
    end

    def self.register(name)
      $mxnet_initializer_registry ||= {}
      $mxnet_initializer_registry[name.to_sym] = self
    end

    def self.create(initializer)
      case initializer
      when ::Class
        initializer.new
      when ::String, ::Symbol
        $mxnet_initializer_registry[initializer.to_sym].new
      else
        initializer
      end
    end

    ##
    # Calls #init_array.
    #
    def [](array)
      init_array(array)
    end

    ##
    # Override to initialize array.
    #
    # ====Parameters
    #
    # +array+:: (NDArray) Array to initialize.
    #
    def init_array(array)
      raise NotImplementedError
    end

    ##
    # Initializes array to zero.
    #
    class Zero < Initializer
      register :zeros
      def init_array(array)
        array[0..-1] = 0.0
      end
    end

    def self.Zero(*args)
      Zero.new(*args)
    end

    ##
    # Initializes array with random values uniformly sampled from a
    # given range.
    #
    class Uniform < Initializer
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +scale+:: (Float, optional)
      #           The bound on the range of the generated random values.
      #           Values are generated from the range <tt>[-scale, scale]</tt>.
      #           Default scale is 0.07.
      #
      def initialize(scale = 0.07)
        @scale = scale
      end

      def init_array(array)
        MXNet::NDArray::Random.uniform(-@scale, @scale, out: array)
      end
    end

    def self.Uniform(*args)
      Uniform.new(*args)
    end

    ##
    # Initializes array with random values sampled from a normal
    # distribution with a mean of zero and standard deviation of
    # +sigma+.
    #
    class Normal < Initializer
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +sigma+:: (Float, optional)
      #           Standard deviation of the normal distribution.
      #           Default standard deviation is 0.01.
      #
      def initialize(sigma = 0.01)
        @sigma = sigma
      end

      def init_array(array)
        MXNet::NDArray::Random.normal(0, @sigma, out: array)
      end
    end

    def self.Normal(*args)
      Normal.new(*args)
    end
  end
end
