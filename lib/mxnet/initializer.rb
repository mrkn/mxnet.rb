require 'mxnet/registry'
require 'json'

module MXNet
  module Init
    # Descriptor for the initialization pattern
    class InitDesc < String
      # @param name [String] Name of variable
      # @param attrs [Hash{Symbol => String}]
      #   Attributes of this variable taken from `MXNet::Symbol.attr_dict`
      # @param global_init [MXNet::Init::Initializer]
      #   Global initializer to fallback to
      def initialize(name, attrs=nil, global_init=nil)
        if name.is_a? ::Symbol
          name = name.to_s
        elsif name.respond_to? :to_str
          name = name.to_str
        end
        super(name)
        @attrs = attrs || {}
        @global_init = global_init
      end

      attr_accessor :attrs, :global_init
    end

    # The base class of initializers.
    class Initializer
      def initialize(**kwargs)
        @kwargs = kwargs
        @verbose = false
        @print_func = nil
      end

      # Switch on/off verbose mode
      #
      # @param verbose [true,false] switch on/off verbose mode
      # @param print_func [Proc] A block that computes statistics of
      #   initialized arrays.   Takes an `NDArray` and returns an `str`.
      #   Defaults to mean absolute value `(|x| / x.size).as_scalar.to_s`.
      def set_verbosity(verbose=false, &print_func)
        @verbose = verbose
        unless print_func
          print_func = lambda do |x|
            (NDArray.norm(x) / Math.sqrt(x.size)).as_scalar.to_s
          end
        end
        @print_func = print_func
        self
      end

      # Internal verbose print function
      #
      # @param desc [MXNet::Init::InitDesc, String] name of the array
      # @param init [String] initializer pattern
      # @param arr [MXNet::NDArray] initialized array
      private def verbose_print(desc, init, arr)
        if @verbose && @print_func
          # TODO:
          # logging.info("Initialized #{desc} as #{init}: #{@print_func.(arr)}")
        end
      end

      # Saves the initializer to string
      #
      # @return [String] JSON formatted string that describes the initializer.
      #
      # == Example:
      #
      #     > init = MXNet::Init::Normal.new(0.5)
      #     > init.dumps
      #     '["normal", { "sigma": 0.5 }]'
      #     > init = MXNet::Init::Xavier.new(factor_type: :in, magnitude: 2.34)
      #     > init.dumps
      #     '["xavier", {"rnd_type": "uniform", "magnitude": 2.34, "factor_type": "in"}]'
      def dumps
        JSON.dump([self.class.name[/::([^:]+)\z/, 1].downcase, @kwargs])
      end

      # Initialize an array
      #
      # @param desc [MXNet::Init::InitDesc] Initialization pattern descriptor
      # @param arr [MXNet::NDArray] The array to be initialized
      def call(desc, arr)
        unless desc.is_a? InitDesc
          raise TypeError, "desc should be a MXNet::Init::InitDesc"
        end

        desc.global_init ||= self
        init = desc.attrs[:__init__]
        if init
          MXNet::Init.registry_manager.create(init).send :init_weight, desc, arr
          verbose_print(desc, init, arr)
        else
          # register nnvm::FSetInputVariableAttrs in the backend for new patterns
          # don't add new cases here.
          case
          when desc.end_with?('weight')
            init_weight(desc, arr)
            verbose_print(desc, 'weight', arr)
          when desc.end_with?('bias')
            init_bias(desc, arr)
            verbose_print(desc, 'bias', arr)
          when desc.end_with?('gamma')
            init_gamma(desc, arr)
            verbose_print(desc, 'gamma', arr)
          when desc.end_with?('beta')
            init_beta(desc, arr)
            verbose_print(desc, 'beta', arr)
          else
            init_default(desc, arr)
          end
        end
      end

      private def init_bilinear(_, arr)
        n = arr.shape.inject(:*)
        weight = MXNet::NDArray.zeros(n, dtype: :float32)
        shape = arr.shape
        f = (shape[3] / 2.0).ceil
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        0.upto(n - 1) do |i|
          x = i % shape[3]
          y = (i / shape[3]) % shape[2]
          weight[i] = (1 - (x / f - c).abs) * (1 - (y / f - c).abs)
        end
        arr[0..-1] = weight.reshape(shape)
      end

      private def init_loc_bias(_, arr)
        shape = arr.shape
        raise ArgumentError, "arr.shape[0] must eq 6" unless shape[0] == 6
        arr[0..-1] = [1.0, 0, 0, 0, 1.0, 0] # TODO
      end

      private def init_zero(_, arr)
        arr[0..-1] = 0.0
      end

      private def init_one(_, arr)
        arr[0..-1] = 1.0
      end

      private def init_bias(_, arr)
        arr[0..-1] = 0.0
      end

      private def init_gamma(_, arr)
        arr[0..-1] = 1.0
      end

      private def init_beta(_, arr)
        arr[0..-1] = 0.0
      end

      private def init_weight(name, arr)
        raise NotImplementedError, "subclass responsibility"
      end

      private def init_default(name, _)
        raise ArgumentError,
          "Unknown initialization pattern for #{name}. " +
          'Default initialization is now limited to ' +
          '"weight", "bias", "gamma" (1.0), and "beta" (0.0).' +
          'Please use MXNet::Symbol.var(init: MXNet::Init::*) ' +
          'to set initialization pattern'
      end
    end

    def self.registry_manager
      @registry_manager ||= MXNet::Registry::Manager.new(Initializer, :initializer)
    end

    # TODO: Load
    # TODO: Mixed
    # TODO: Zero

    # Initializes weights to one.
    #
    # == Example:
    #
    #     init = MXNet::Init::One.new()
    #     module.init_params(init)
    #     module.each_param do |dict|
    #       dict.each do |key, val|
    #         puts "#{key} => #{val.to_narray}"
    #       end
    #     end
    #     #=> fullyconnected0_weight => [[1, 1, 1, 1]]
    class One < Initializer
      private def init_weight(_, arr)
        arr[0..-1] = 1.0
      end
    end

    # TODO: Constant

    # Initializes weights with random variables uniformly sampled from a given range.
    class Uniform < Initializer
      def initialize(scale: 0.07)
        super(scale: scale)
        @scale = scale
      end

      private def init_weight(_, arr)
        MXNet::NDArray::Random.uniform(-@scale, @scale, out: arr)
      end
    end

    # Initializes weights with random values sampled from a normal distribution
    # with a mean of zero and standard deviation of `sigma`.
    class Normal < Initializer
      # @param sigma [Float] Standard deviation of the normal distribution.
      #   Default standard deviation is 0.01.
      def initialize(sigma: 0.01)
        super(sigma: sigma)
        @sigma = sigma
      end

      private def init_weight(_, arr)
        MXNet::NDArray::Random.Normal(0, @sigma, out: arr)
      end
    end

    # TODO: Orthogonal

    class Xavier < Initializer
      def initialize(rnd_type: :uniform, factor_type: :avg, magnitude: 3)
        super(rnd_type: rnd_type, factor_type: factor_type, magnitude: magnitude)
        @rnd_type = rnd_type
        @factor_type = factor_type
        @magnitude = magnitude
      end

      attr_reader :rnd_type, :factor_type, :magnitude

      private def init_weight(name, arr)
        shape = arr.shape
        hw_scale = 1.0
        if shape.length < 2
          raise ArgumentError, "Xavier initializer cannot be applied to vector #{name}.  It requires at leaset 2D."
        end
        if shape.length > 2
          hw_scale = shape[2..-1].inject(:*)
        end
        fan_in  = shape[1] * hw_scale
        fan_out = shape[0] * hw_scale
        factor = 1.0
        case @factor_type
        when :avg
          factor = (fan_in + fan_out) / 2.0
        when :in
          factor = fan_in
        when :out
          factor = fan_out
        else
          raise ArgumentError, "Incorrect factor type"
        end
        scale = Math.sqrt(@magnitude / factor)
        case @rnd_type
        when :uniform
          MXNet::NDArray.random_uniform(low: -scale, high: scale, out: arr)
        when :gaussian
          MXNet::NDArray.random_normal(loc: 0, scale: scale, out: arr)
        else
          raise ArgumentError, "Unknown random type"
        end
      end
    end

    registry_manager.register Xavier

    # TODO: MSRAPrelu < Xavier
    # TODO: Bilinear
    # TODO: LSTMBias
    # TODO: FusedRNN
  end
end
