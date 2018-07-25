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
        # +batch_axis+:: (integer)
        #                The axis that represents the
        #                mini-batch.
        #
        def initialize(weight:, batch_axis:, **kwargs)
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
        def apply_weighting(clazz, loss, weight = nil, sample_weight = nil)
          unless sample_weight.nil?
            loss = clazz.broadcast_mul(loss, sample_weight)
          end
          unless weight.nil?
            raise ArgumentError, 'weight must be numeric' unless weight.is_a?(Numeric)
            loss = loss * weight
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
        # +weight+::     (float or +nil+, default +nil+)
        #                Global scalar weight for loss.
        # +batch_axis+:: (integer, default 0)
        #                The axis that represents mini-batch.
        #
        def initialize(weight: nil, batch_axis: 0, **kwargs)
          super(weight: weight, batch_axis: batch_axis, **kwargs)
        end

        def hybrid_forward(clazz, prediction, label, sample_weight: nil)
          label = clazz.reshape_like(label, prediction)
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
        # +weight+::     (float or +nil+, default 1.0)
        #                Global scalar weight for loss.
        # +batch_axis+:: (integer, default 0)
        #                The axis that represents mini-batch.
        #
        def initialize(weight: 1.0, batch_axis: 0, **kwargs)
          super(weight: weight, batch_axis: batch_axis, **kwargs)
        end

        def hybrid_forward(clazz, prediction = nil, label = nil, sample_weight: nil, **kwargs)
          prediction ||= kwargs[:prediction]
          label ||= kwargs[:label]
          label = clazz.reshape_like(label, prediction)
          loss = clazz.square(prediction - label)
          loss = apply_weighting(clazz, loss, @weight/2, sample_weight)
          clazz.mean(loss, axis: @batch_axis, exclude: true)
        end
      end

      ##
      # Computes the softmax cross-entropy loss.
      #
      # If +sparse_label+ is +true+ (default), labels should contain
      # integer category indicators. The labels' shape should be the
      # predictions' shape with the +axis+ dimension removed -- i.e. for
      # predictions with shape <tt>[1, 2, 3, 4]</tt> and <tt>axis: 2</tt>,
      # labels' shape should be <tt>[1, 2, 4]</tt>.
      #
      # If +sparse_label+ is +false+, labels should contain probability
      # distributions and labels' shape should be the same as
      # predictions' shape.
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
        #                  Whether prediction is a log probability
        #                  (usually from #log_softmax) instead of
        #                  unnormalized numbers.
        # +weight+::       (float or +nil+, default +nil+)
        #                  Global scalar weight for loss.
        # +batch_axis+::   (integer, default 0)
        #                  The axis that represents mini-batch.
        #
        def initialize(axis: -1, sparse_label: true, from_logits: false,
                       weight: nil, batch_axis: 0, **kwargs)
          @axis = axis
          @sparse_label = sparse_label
          @from_logits = from_logits
          super(weight: weight, batch_axis: batch_axis, **kwargs)
        end

        def hybrid_forward(clazz, prediction, label, sample_weight: nil)
          unless @from_logits
            prediction = clazz.log_softmax(prediction, axis: @axis)
          end
          if @sparse_label
            loss = -clazz.pick(prediction, label, axis: @axis, keepdims: true)
          else
            label = clazz.reshape_like(label, prediction)
            loss = -clazz.sum(prediction * label, axis: @axis, keepdims: true)
          end
          loss = apply_weighting(clazz, loss, @weight, sample_weight)
          clazz.mean(loss, axis: @batch_axis, exclude: true)
        end
      end
    end
  end
end
