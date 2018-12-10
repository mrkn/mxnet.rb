require 'mxnet/gluon/block'

module MXNet::Gluon
  ##
  # Base class for loss.
  #
  class Loss < MXNet::Gluon::HybridBlock
    def initialize(**kwargs)
      super(**kwargs)
    end

    ##
    # Override to construct symbolic graph for this Block.
    #
    # ====Parameters
    #
    # +args+:: (array of NDArray or Symbol) Input tensors.
    #
    #
    def hybrid_forward(clazz, *args)
      raise NotImplementedError
    end

    protected

    ##
    # Reshapes x to be the same shape as y.
    #
    def reshape_like(clazz, x, y)
      case clazz
      when MXNet::NDArray
        x.reshape(y.shape)
      else
        clazz.reshape_like(x, y)
      end
    end

    ##
    # Apply weighting to loss.
    #
    # ====Parameters
    #
    # +loss+::          (Symbol or NDArray)
    #                   The loss to be weighted.
    # +weight+::        (float or +nil+)
    #                   Global scalar weight for loss.
    # +sample_weight+:: (Symbol, NDArray or +nil+)
    #                   Per sample weighting. Must be broadcastable to
    #                   the same shape as loss. For example, if loss
    #                   has shape (64, 10) and you want to weight each
    #                   sample in the batch separately, +sample_weight+
    #                   should have shape (64, 1).
    #
    # ====Returns
    #
    # Weighted loss
    #
    #
    def apply_weighting(clazz, loss, weight = nil, sample_weight = nil)
      unless weight.nil?
        raise ArgumentError, 'weight must be numeric' unless weight.is_a?(Numeric)
        loss = loss * weight
      end
      unless sample_weight.nil?
        loss = clazz.broadcast_mul(loss, sample_weight)
      end
      loss
    end

    ##
    # Calculates the mean squared error between prediction and label.
    #
    # Inputs "prediction" and "label" can have arbitrary shape as long
    # as they have the same number of elements.
    #
    class L2Loss < Loss
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +weight+::     (float or +nil+)
      #                Global scalar weight for loss.
      # +batch_axis+:: (integer, default 0)
      #                The axis that represents mini-batch.
      #
      def initialize(weight: 1.0, batch_axis: 0, **kwargs)
        super(**kwargs)
        @weight = weight
        @batch_axis = batch_axis
      end

      def hybrid_forward(clazz, prediction = nil, label = nil, sample_weight: nil, **kwargs)
        prediction ||= kwargs[:prediction]
        label ||= kwargs[:label]
        label = reshape_like(clazz, label, prediction)
        loss = clazz.square(prediction - label)
        loss = apply_weighting(clazz, loss, @weight/2, sample_weight)
        clazz.mean(loss, axis: @batch_axis, exclude: true)
      end
    end

    def self.L2Loss(*args)
      L2Loss.new(*args)
    end
  end
end
