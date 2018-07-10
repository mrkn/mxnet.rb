require 'mxnet/gluon'

module MXNet::Gluon
  ##
  # A Container holding parameters (weights) of Blocks.
  #
  # ====Parameters
  #
  # +name+::  (string) Name of this parameter.
  # +shape+:: (integer or array of integers) Shape of this parameter.
  #           By default shape is not specified.
  #
  class Parameter
    def initialize(name, shape: nil, dtype: 0)
      @name = name
      shape = [shape] if shape.is_a?(Integer)
      self.shape = shape
      @dtype = dtype
      @data = nil
      @grad = nil
    end
    def name
      @name
    end
    def shape
      @shape
    end
    def shape=(shape)
      @shape ||= shape
    end
    ##
    # Initializes parameter and gradient arrays. Only used for NDArray
    # API.
    #
    # ====Parameters
    #
    # +ctx+::          (Context or array of Contexts)
    #                  Desired contexts. Initialize Parameter on
    #                  given contexts.
    # +default_init+:: (Initializer, default MXNet::Uniform)
    #                  Default initializer.
    #
    def init(ctx: nil, default_init: MXNet::Uniform)
      @data = @grad = nil
      ctx = [MXNet.current_context] if ctx.nil?
      ctx = [ctx] if ctx.is_a?(MXNet::Context)
      data = MXNet::NDArray.zeros(@shape, dtype: @dtype, ctx: MXNet.cpu)
      default_init.new[data]
      @data = ctx.map { |c| data.copy_to(c) }
    end
    ##
    # Returns a copy of this parameter on one context. Must have been
    # initialized on this context before.
    #
    # ====Parameters
    #
    # +ctx+:: (Context) Desired context.
    #
    # ====Returns
    #
    # NDArray on context.
    #
    def data(ctx: nil)
      check_and_get(@data, ctx)
    end
    def to_s
      "Parameter #{@name} (shape=#{@shape.inspect}, dtype=#{@dtype.inspect})"
    end
    def ==(other)
      self.name == other.name && self.shape == other.shape
    rescue NoMethodError
      false
    end
    private
    def check_and_get(arr_list, ctx)
      unless arr_list.nil?
        if ctx.nil?
          if arr_list.length == 1
            return arr_list[0]
          end
          ctx = MXNet.current_context
        end
        if data = arr_list.find { |arr| arr.context == ctx }
          return data
        end
        raise RuntimeError.new("Parameter '#{self.name}' was not initialized on context #{ctx}.")
      end
      raise RuntimeError.new("Parameter '#{self.name}' has not been initialized. You should initialize parameters.")
    end
  end
  ##
  # A dictionary managing a set of Parameters.
  #
  # ====Parameters
  #
  # +prefix+:: (string, default "") The prefix to be prepended to all
  #            Parameters' names created by this dict.
  # +shared+:: (ParameterDict or +nil+) If not +nil+, when this dict's
  #            #get method creates a new parameter, will first try to
  #            retrieve it from "shared" dict. Usually used for
  #            sharing parameters with another Block.
  #
  class ParameterDict
    include Enumerable
    def initialize(prefix: '', shared: nil)
      @params = {}
      @prefix = prefix
      @shared = shared
    end
    def each(&block)
      @params.each(&block)
    end
    ##
    # Prefix of this dict. It will be prepended to a Parameter's name
    # created with #get.
    #
    def prefix
      @prefix
    end
    ##
    # Retrieves a Parameter with name "<prefix><name>". If not found,
    # #get will first try to retrieve it from "shared" dict. If still
    # not found, #get will create a new Parameter with key-word
    # arguments and both store and return it.
    #
    # ====Parameters
    #
    # +name+:: (string) Name of the desired Parameter. It will be
    #          prepended with this dict's prefix.
    #
    # ====Returns
    #
    # The created or retrieved Parameter.
    #
    def get(name, **kwargs)
      name = @prefix + name
      unless param = _get(name)
        param = @params[name] = Parameter.new(name, **kwargs)
      else
        param
      end
    end
    ##
    # Copies all Parameters in +other+ to this dict.
    #
    def update(other)
      other.each do |k, v|
        if @params[k] && @params[k] != v
          raise ArgumentError, "Cannot update because keys have different values: #{@params[k]}, #{v}"
        end
      end
      other.each do |k, v|
        @params[k] = v
      end
    end
    ##
    # Initializes all Parameters managed by this dict to be used for
    # NDArray API. It has no effect when using Symbol API.
    #
    # +ctx+:: (Context or array of Context)
    #         Desired contexts. Initialize Parameter on
    #         given contexts.
    #
    def init(ctx: nil)
      self.each { |_, v| v.init(ctx: ctx) }
    end
    def to_s
      "ParameterDict (\n" +
        self.inject('') { |a, (n, p)| a + "  #{p}\n" } +
        ')'
    end
    def ==(other)
      self.prefix == other.prefix && self.each.to_a == other.each.to_a
    rescue NoMethodError
      false
    end
    protected
    def _get(name)
      if value = @params[name]
        value
      elsif @shared && value = @shared._get(name)
        @params[name] = value
        value
      end
    end
  end
end
