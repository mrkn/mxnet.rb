require 'mxnet'

module MXNet
  ##
  # The base class of an initializer.
  #
  class Initializer
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
  end
  ##
  # Initializes weights with random values uniformly sampled from a
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
  ##
  # Initializes weights with random values sampled from a normal
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
end
