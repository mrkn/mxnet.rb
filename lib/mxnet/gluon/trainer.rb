require 'mxnet/gluon'
require 'mxnet/optimizer'

module MXNet
  module Gluon
    ##
    # Applies an Optimizer on a set of Parameters. Trainer should be
    # used together with +autograd+.
    #
    class Trainer
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +params+::           (ParameterDict)
      #                      The set of parameters to optimize.
      # +optimizer+::        (Optimizer)
      #                      The optimizer to use.
      # +optimizer_params+:: (Hash)
      #                      Key-word arguments to be passed to
      #                      optimizer constructor. For example,
      #                      <tt>{'learning_rate': 0.1}</tt>. See each
      #                      optimizer's constructor for a list of
      #                      additional supported arguments.
      #
      def initialize(params, optimizer, optimizer_params = {})
        if params.is_a?(Hash) || params.is_a?(ParameterDict)
          params = params.values
        end
        @params = []
        params.each.with_index do |param, i|
          unless param.is_a?(Parameter)
            raise ArgumentError, 'First argument must be a Hash or ParameterDict.'
          end
          param.trainer = self
          @params << param
        end
        check_contexts
        init_optimizer(optimizer, optimizer_params)
        @scale = optimizer_params[:rescale_grad] || 1.0
      end

      ##
      # Makes one step of parameter update.
      #
      # ====Parameters
      #
      # +batch_size+:: (integer)
      #                Batch size of data processed. Gradient will be
      #                normalized by `1/batch_size`. Set this to 1 if
      #                you normalized loss manually with `loss =
      #                mean(loss)`.
      #
      def step(batch_size)
        @optimizer.rescale_grad = @scale / batch_size
        _update
      end

      ##
      # Makes one step of parameter update.
      #
      # ====Parameter
      #
      # +batch_size+:: (integer)
      #                Batch size of data processed. Gradient will be
      #                normalized by `1/batch_size`. Set this to 1 if
      #                you normalized loss manually with `loss =
      #                mean(loss)`.
      #
      def update(batch_size)
        @optimizer.rescale_grad = @scale / batch_size
        _update
      end

      private

      def check_contexts
        contexts = nil
        @params.each do |param|
          unless contexts.nil? || contexts == param.list_ctx
            raise RuntimeError,
                  "All Parameters must be initialized on the same set of contexts, " \
                  "but Parameter '#{param.name}' is initialized on '#{param.list_ctx.map(&:to_s)}' " \
                  "while previous Parameters are initialized on '#{contexts.map(&:to_s)}'."
          end
          contexts = param.list_ctx
        end
        @contexts = contexts
      end

      def init_optimizer(optimizer, optimizer_params)
        @optimizer = MXNet::Optimizer.create(optimizer, **optimizer_params)
      end

      def _update
        @params.each.with_index do |param, i|
          param.list_data.zip(param.list_grad) do |data, grad|
            @optimizer.update(i, data, grad, nil)
          end
        end
      end
    end

    def self.Trainer(*args)
      Trainer.new(*args)
    end
  end
end
