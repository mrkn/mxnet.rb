require 'mxnet/gluon'

module MXNet
  module Gluon
    module Loss
      ##
      # Base class for loss.
      #
      class Base < MXNet::Gluon::HybridBlock
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
	  @weight = weight
	  @batch_axis = batch_axis
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
      end

      ##
      # Calculates the mean absolute error between prediction and label.
      #
      # Inputs "prediction" and "label" can have arbitrary shape as long
      # as they have the same number of elements.
      #
      class L1Loss < Base
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
        def initialize(**kwargs)
          super(**kwargs)
        end

        def hybrid_forward(clazz, prediction, label, sample_weight: nil)
          label = reshape_like(clazz, label, prediction)
          loss = clazz.abs(prediction - label)
          loss = apply_weighting(clazz, loss, @weight, sample_weight)
          clazz.mean(loss, axis: @batch_axis, exclude: true)
        end
      end

      ##
      # Calculates the mean squared error between prediction and label.
      #
      # Inputs "prediction" and "label" can have arbitrary shape as long
      # as they have the same number of elements.
      #
      class L2Loss < Base
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
        def initialize(**kwargs)
          super(**kwargs)
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

      ##
      # Computes the softmax cross-entropy loss.
      #
      class SoftmaxCrossEntropyLoss < Base
        ##
        # Creates a new instance.
        #
        # ====Parameters
        #
        # +axis+::         (integer, default -1)
        #                  The axis to sum over when computing softmax
        #                  and entropy.
        # +sparse_label+:: (boolean, default +true+)
        #                  Whether label is an integer array instead of
        #                  probability distribution.
        # +from_logits+::  (boolean, default +false+)
        #                  Whether input is a log probability (usually
        #                  from #log_softmax) instead of unnormalized
        #                  numbers.
        # +weight+::       (float or +nil+)
        #                  Global scalar weight for loss.
        # +batch_axis+::   (integer, default 0)
        #                  The axis that represents mini-batch.
        #
        def initialize(axis: -1, sparse_label: true, from_logits: false, **kwargs)
          @axis = axis
          @sparse_label = sparse_label
          @from_logits = from_logits
          super(**kwargs)
        end

        def hybrid_forward(clazz, prediction, label, sample_weight: nil)
          unless @from_logits
            prediction = clazz.log_softmax(prediction, axis: @axis)
          end
          if @sparse_label
            loss = -clazz.pick(prediction, label, axis: @axis, keepdims: true)
          else
            label = reshape_like(clazz, label, prediction)
            loss = -clazz.sum(prediction * label, axis: @axis, keepdims: true)
          end
          loss = apply_weighting(clazz, loss, @weight, sample_weight)
          clazz.mean(loss, axis: @batch_axis, exclude: true)
        end
      end
    end
  end
end
