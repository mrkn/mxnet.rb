require 'mxnet'

module MXNet
  ##
  # The base class inherited by all optimizers.
  #
  # Custom optimizers can be created by subclassing Optimizer and
  # implementing the required function #update. By default, the
  # created optimizer will be registered under its simplified class
  # name (`class.name.split('::').last.downcase.to_sym`) but it may be
  # registered under another name by calling #register.
  #
  #     class MyOptimizer < Optimizer
  #       register :myopt
  #       def update(index, weight, gradient, state)
  #         ...
  #       end
  #     end
  #
  class Optimizer
    def self.inherited(child)
      default = child.name.split('::').last.downcase
      child.register(default)
    end

    def self.register(name)
      $mxnet_optimizer_registry ||= {}
      $mxnet_optimizer_registry[name.to_sym] = self
    end

    def self.create(optimizer, **kwargs)
      case optimizer
      when ::Class
        optimizer.new(**kwargs)
      when ::String, ::Symbol
        $mxnet_optimizer_registry[optimizer.to_sym].new(**kwargs)
      else
        optimizer
      end
    end

    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +rescale_grad+::  (float, optional)
    #                   Before updating, multiply the gradient with
    #                   "rescale_grad". Often choose to be
    #                   <tt>1.0/batch_size</tt>.
    # +learning_rate+:: (float, optional)
    #                   The initial learning rate.
    # +wd+::            (float, optional)
    #                   The weight decay (or L2 regularization)
    #                   coefficient. Modifies objective by adding a
    #                   penalty for having large weights.
    #
    def initialize(rescale_grad: 1.0, learning_rate: 0.01, wd: 0.0)
      @rescale_grad = rescale_grad
      @lr = learning_rate
      @wd = wd
    end

    attr_accessor :rescale_grad

    ##
    # Updates the given parameter using the corresponding gradient and
    # state.
    #
    # ====Parameters
    #
    # +index+::    (integer)
    #              The unique index of the parameter into the
    #              individual learning rates and weight
    #              decays. Learning rates and weight decay may be set
    #              via #set_lr_mult and #set_wd_mult, respectively.
    # +weight+::   (NDArray)
    #              The parameter to be updated.
    # +gradient+:: (NDArray)
    #              The gradient of the objective with respect to this
    #              parameter.
    # +state+::    (any)
    #              The state returned by #create_state.
    #
    def update(index, weight, gradient, state)
      raise NotImplementedError
    end

    ##
    # The SGD optimizer with momentum and weight decay.
    #
    class SGD < Optimizer
      ##
      # Creates a new instance.
      #
      # This optimizer accepts the following parameters in addition to
      # those accepted by Optimizer.
      #
      # ====Parameters
      #
      # +momentum+:: (float, optional)
      #              The momentum value.
      #
      def initialize(momentum: 0.0, **kwargs)
        super(**kwargs)
        @momentum = momentum
      end

      def update(index, weight, gradient, state)
        lr = get_lr(index)
        wd = get_wd(index)
        kwargs = {}.tap do |kwargs|
          kwargs[:momentum] = @momentum if @momentum > 0
          kwargs[:rescale_grad] = @rescale_grad
        end
        MXNet::NDArray.sgd_update(weight, gradient, out: weight, lr: lr, wd: wd, **kwargs)
      end
    end

    def self.SGD(*args)
      SGD.new(*args)
    end

    protected

    ##
    # Gets the learning rate given the index of the weight.
    #
    # ====Parameters
    #
    # +index+:: (integer)
    #           The index corresponding to the weight.
    #
    # ====Returns
    #
    # Learning rate for this index.
    #
    def get_lr(index)
      @lr
    end

    ##
    # Gets weight decay for index.
    #
    # ====Parameters
    #
    # +index+:: (integer)
    #           The index corresponding to the weight.
    #
    # ====Returns
    #
    # Weight decay for this index.
    #
    def get_wd(index)
      @wd
    end
  end
end
