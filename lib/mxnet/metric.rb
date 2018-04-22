require 'mxnet/registry'
module MXNet
  module Metric
    # Base class for all evaluation metrics.
    class EvalMetric
      def initialize(name, output_names: nil, label_names: nil, **kwargs)
        @name = case name
                when Symbol, String
                  name.to_sym
                else
                  name.to_s.to_sym
                end
        @output_names = output_names
        @label_names = label_names
        @kwargs = kwargs
        reset
      end

      attr_reader :name, :output_names, :label_names

      # TODO: to_s
      # TODO: get config
      # TODO: update_dict

      # Updates the internal evaluation resut.
      def update(labels, preds)
        raise NotImplementedError
      end

      # Resets the internal evaluation result to initial state.
      def reset
        @num_inst = 0
        @sum_metric = 0.0
        nil
      end

      # Get the current evaluation result.
      def get
        if @num_inst == 0
          return [@name, Float::NAN]
        else
          return [@name, @sum_metric / @num_inst]
        end
      end

      # Returns zipped name and value pairs.
      def get_name_value
        name, value = get
        name = [name] unless name.is_a? Array
        value = [value] unless value.is_a? Array
        return name.zip(value)
      end

      # Helper function for checking shape of label and prediction
      private def check_label_shapes(labels, preds, wrap=false, shape=false)
        if !shape
          labels_shape = labels.is_a?(MXNet::NDArray) ? labels.shape[0] : labels.length
          preds_shape = preds.is_a?(MXNet::NDArray) ? preds.shape[0] : preds.length
        else
          label_shape, pred_shape = labels.shape, preds.shape
        end

        if label_shape != pred_shape
          raise ArgumentError, "Shape of labels #{label_shape} does not " +
                "match shape of predictions #{pred_shape}"
        end

        if wrap
          labels = [labels] if labels.is_a?(MXNet::NDArray)
          preds = [preds] if preds.is_a?(MXNet::NDArray)
        end

        return [labels, preds]
      end
    end

    def self.registry_manager
      @registry_manager ||= Registry::Manager.new(EvalMetric, :metric)
    end
    # Manages multiple evaluation metrics.
    class CompositeEvalMetric < EvalMetric
      def initialize(metrics=nil, name: :composite, output_names: nil, label_names: nil)
        super(name, output_names: output_names, label_names: label_names)
        @metrics = (metrics || []).map {|m| Metric.registry_manager.create(m) }
      end

      # Adds a child metric.
      def add(metric)
        @metrics << create(metric) # TODO: create of repository manager
        self
      end

      alias_method :<<, :add

      # Returns a child metric.
      def get_metric(index)
        case index
        when -@metrics.length ... @metrics.length
          return @metrics[index]
        else
          raise ArgumentError, "Metric index #{index} is out of range 0 and #{@metrics.length}"
        end
      end

      # TODO: def update_dict

      def update(labels, preds)
        @metrics.each do |metric|
          metric.update(labels, preds)
        end
      end

      def reset
        return unless @metrics
        @metrics.each do |metric|
          metric.reset
        end
        nil
      end

      def get
        names = []
        values = []
        @metrics.each do |metric|
          name, value = metric.get
          name = name.to_sym if name.is_a?(String)
          names.concat Array(name)
          values.concat Array(value)
        end
        return [names, values]
      end

      def get_config
        config = super
        config.update(metrics: @metrics.map(&:get_config))
        return config
      end
    end

    registry_manager.register(CompositeEvalMetric)
    registry_manager.alias(CompositeEvalMetric, :composite)

    # CLASSIFICATION METRICS

    # Computes accuracy classification score.
    class Accuracy < EvalMetric
      def initialize(axis: 1, name: :accuracy, output_names: nil, label_names: nil)
        super(name, axis: axis, output_names: output_names, label_names: label_names)
        @axis = axis
      end

      def update(labels, preds)
        labels, preds = check_label_shapes(labels, preds, true)

        labels.each_with_index do |label, index|
          pred_label = preds[index]
          if pred_label.shape != label.shape
            pred_label = pred_label.argmax(axis: @axis)
          end
          pred_label = pred_label.as_in_context(MXNet.cpu).as_type(:int32)
          label = label.as_in_context(MXNet.cpu).as_type(:int32)
          # flatten before checking shapes to avoid shape miss match
          label = label.reshape([-1])
          pred_label = pred_label.reshape([-1])

          labels, preds = check_label_shapes(label, pred_label)

          @sum_metric += (pred_label == label).sum.as_scalar
          @num_inst += pred_label.length
        end

        nil
      end
    end

    registry_manager.register(Accuracy)
    registry_manager.alias(Accuracy, :acc)

    class TopKAccuracy < EvalMetric
      def initialize(top_k: 1, name: :top_k_accuracy, output_names: nil, label_names: nil)
        super(name, top_k: top_k, output_names: output_names, label_names: label_names)
        raise ArgumentError, "Use Accuracy for top_k == 1" unless top_k > 1
        @top_k = top_k
        @name = :"#{self.name}_#{@top_k}"
      end

      def update(labels, preds)
        labels, preds = check_label_shapes(labels, preds, true)
        labels.each_with_index do |label, index|
          pred_label = preds[index]
          if pred_label.shape.length > 2
            raise ArgumentError, "Predictions should be no more than 2 dims"
          end
          pred_label = pred_label.as_in_context(MXNet.cpu).as_type(:float32).argsort(axis: 1)
          label = label.as_in_context(MXNet.cpu).as_type(:int32)
          check_label_shapes(label, pred_label)
          num_samples = pred_label.shape[0]
          case pred_label.shape.length
          when 1
            @sum_metric += (pred_label.reshape([-1]) == label.reshape([-1])).sum.as_scalar
          when 2
            num_classes = pred_label.shape[1]
            top_k = [num_classes, @top_k].min
            top_k.times do |j|
              @sum_metric += (
                pred_label[0..-1, num_classes - 1 - j].reshape([-1]) == label.reshape([-1]).as_type(:float32)
              ).sum.as_scalar
            end
          end
          @num_inst += num_samples
        end
      end
    end

    registry_manager.register(TopKAccuracy)
    registry_manager.alias(TopKAccuracy, :top_k_accuracy, :top_k_acc)
  end
end
