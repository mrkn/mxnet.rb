module MXNet
  class Context
    DEVICE_TYPE_NAME_FROM_ID = { 1 => :cpu, 2 => :gpu, 3 => :cpu_pinned }.freeze

    DEVICE_TYPE_ID_FROM_NAME = { cpu: 1, gpu: 2, cpu_pinned: 3 }.freeze

    def self.device_type_id_from_name(device_name)
      device_name = device_name.to_sym if device_name.kind_of? String
      DEVICE_TYPE_ID_FROM_NAME[device_name]
    end

    def initialize(device_type, device_id=0)
      if device_type.kind_of? Context
        device_type = device_type.device_type_id
        device_id = device_type.device_id
      end
      case device_type
      when String, ::Symbol
        @device_type_id = Context.device_type_id_from_name(device_type)
      when Integer
        @device_type_id = device_type
      else
        raise ArgumentError,
          "Invalid type of device_type: #{device_type.class} " +
          "for MXNet::Context, String, Symbol, or Integer"
      end
      @device_id = device_id
    end

    attr_reader :device_type_id
    attr_reader :device_id

    def device_type
      DEVICE_TYPE_NAME_FROM_ID[device_type_id]
    end

    def hash
      [device_type_id, device_id].hash
    end

    def eql?(other)
      return false unless other.kind_of? Context
      return device_type_id == other.device_type_id &&
             device_id == other.device_id
    end

    alias_method :==, :eql?

    def to_s
      "#{device_type}(#{device_id})"
    end

    def self.default
      @default ||= Context.new(:cpu, 0)
    end

    class << self
      alias current default
    end

    def self.with(ctx)
      return unless block_given?
      begin
        old_context = @default
        yield
      ensure
        @default = @old_context
      end
    end
  end

  def self.cpu(device_id=0)
    Context.new(:cpu, device_id)
  end

  def self.gpu(device_id=0)
    Context.new(:gpu, device_id)
  end

  def self.current_context
    Context.default
  end
end
