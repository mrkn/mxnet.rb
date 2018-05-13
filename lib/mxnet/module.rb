module MXNet
  # The base class of a module.
  #
  # ...
  class BaseModule
    def initialize(logger: nil) # TODO: logger
      @logger = logger
      @binded = false
      @for_training = false
      @inputs_need_grad = false
      @params_initialized = false
      @optimizer_initialized = false
      @symbol = nil
      @total_exec_bytes = 0
    end
  end

  # MXNet::Module is a basic module that wrap a `Symbol`.
  class Module < BaseModule
    def initialize(symbol, data_names: [:data], label_names: [:softmax_label],
                   logger: nil, # TODO: logger
                   context: MXNet.cpu, work_load_list: nil,
                   fixed_param_names: nil, state_names: nil, group2ctxs: nil,
                   compression_params: nil)
      super(logger: logger)
    end
  end
end
