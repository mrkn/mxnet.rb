module MXNet
  module LRScheduler
    class Base
      def initialize(base_lr: 0.01)
        @base_lr = base_lr
      end

      attr_accessor :base_lr

      # Return a new learning rate.
      #
      # The `num_update` is the upper bound of the number of updates applied
      # to every weight.
      #
      # Assume the optimizer has updated *i*-th weight by *k_i* times, namely
      # `optimizer.update(i, weight_i)` is called by *k_i* times.  Then::
      #
      #     num_update = max([k_i for all i])
      #
      # @param num_update [Integer]
      #     The maximum number of updates applied to a weight.
      def call(num_update)
        raise NotImplementedError, "must override this"
      end
    end

    class FactorScheduler < Base
      def initialize(step:, factor: 1, stop_factor_lr: 1e-8)
        super()
        raise ArgumentError, "Schedule step must be greater or equal than 1 round" if step < 1
        raise ArgumentError, "Factor must be no more than 1 to make lr reduce" if factor > 1.0
        @step = step
        @factor = factor
        @stop_factor_lr = stop_factor_lr
        @count = 0
      end

      attr_reader :step, :factor, :stop_factor_lr, :count

      def call(num_update)
        while num_update > @count + @step
          @count += @step
          @base_lr *= @factor
          if @base_lr < @stop_factor_lr
            @base_lr = @stop_factor_lr
            # logging.info(("Update[%d]: now learning rate arrived at %0.5e, " +
            #               "will not change in the future") % [num_update, @base_lr])
          else
            # logging.info("Update[%d]: Change learning rate to %0.5e" % [num_update, @base_lr])
          end
        end
        return @base_lr
      end

      alias_method :update, :call
    end

    # Reduce the learning rate by given a list of steps.
    class MultiFactorScheduler < Base
      def initialize(step:, factor: 1)
        super()
        unless step.is_a?(Array) && step.length > 1
          raise ArgumentError, "Schedule step must be an Array with more than 1 items"
        end
        step.each_cons(2) do |s0, s1|
          raise ArgumentError, "Schedule step must be an increasing integer array" if s1 <= s0
          raise ArgumentError, "Schedule step must be greater than or equal to 1" if s0 < 1 || s1 < 1
        end
        raise ArgumentError, "Factor must be no more than 1 to make lr reduce" if factor > 1
        @step = step
        @cur_step_ind = 0
        @factor = factor
        @count = 0
      end

      attr_reader :step, :cur_step_ind, :factor, :count

      def call(num_update)
        while @cur_step_ind < @step.length
          if num_update > @step[@cur_step_ind]
            @count = @step[@cur_step_ind]
            @cur_step_ind += 1
            @base_lr *= @factor
            puts("Update[%d]: Change learning rate to %0.5e" % [num_update, @base_lr])
          else
            return @base_lr
          end
        end
        return @base_lr
      end

      alias_method :update, :call
    end

    # TODO: class PolyScheduler < Base
  end
end
