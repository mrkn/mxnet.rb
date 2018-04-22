require 'spec_helper'
require 'mxnet/metric'

RSpec.describe MXNet::Metric::Accuracy do
  specify do
    predicts = [MXNet::NDArray.array([[0.3, 0.7], [0, 1], [0.4, 0.6]])]
    labels = [MXNet::NDArray.array([0, 1, 1])]
    acc = MXNet::Metric::Accuracy.new
    acc.update(labels, predicts)
    expect(acc.get).to match([:accuracy, a_value_within(1e-15).of(2/3r)])
  end
end

RSpec.describe MXNet::Metric::TopKAccuracy do
  specify do
    MXNet::Random.seed(999)
    top_k = 3
    labels = [MXNet::NDArray.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    predicts = [MXNet::NDArray::Random.uniform(shape: [10, 10])]
    acc = MXNet::Metric::TopKAccuracy.new(top_k: top_k)
    acc.update(labels, predicts)
    expect(acc.get).to match([:top_k_accuracy_3, a_value_within(1e-12).of(0.3)])
  end
end

RSpec.describe MXNet::Metric::CompositeEvalMetric do
  specify do
    MXNet::Random.seed(999)
    top_k = 3
    labels = [MXNet::NDArray.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    predicts = [MXNet::NDArray::Random.uniform(shape: [10, 10])]
    eval_metrics_1 = MXNet::Metric::Accuracy.new
    eval_metrics_2 = MXNet::Metric::TopKAccuracy.new(top_k: top_k)
    eval_metrics = MXNet::Metric::CompositeEvalMetric.new([eval_metrics_1, eval_metrics_2])
    eval_metrics.update(labels, predicts)
    name, acc = eval_metrics.get
    expect(name).to match([:accuracy, :top_k_accuracy_3])
    expect(acc).to match([a_value_within(1e-15).of(0.2),
                          a_value_within(1e-15).of(0.3)])
  end
end
