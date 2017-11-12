module MXNet
  class Executor
    include HandleWrapper

    def initialize(handle, symbol, ctx, grad_req, group2ctx)
      super(handle)
      @arg_arrays = []
      @grad_arrays = []
      @aux_arrays = []
      @outputs = get_outputs
      @symbol = symbol.dup
      @arg_dict = nil
      @grad_dict = nil
      @aux_dict = nil
      @output_dict = nil
      @monitor_callback = nil
      @ctx = ctx.dup
      @grad_req = grad_req.dup
      @group2ctx = group2ctx.dup
    end

    attr_reader :outputs
  end
end
