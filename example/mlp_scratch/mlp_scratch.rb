require 'mxnet'

module MLPScratch
  ND = MXNet::NDArray

  class MLP
    def initialize(num_inputs: 784, num_outputs: 10, num_hidden_units: [256, 128, 64], ctx: nil)
      @layer_dims = [num_inputs, *num_hidden_units, num_outputs]
      @weight_scale = 0.01
      @ctx = ctx || MXNet::Context.default
      @all_parameters = init_parameters
    end

    attr_reader :ctx, :all_parameters, :layer_dims

    private def rnorm(shape)
      ND.random_normal(shape: shape, scale: @weight_scale, ctx: @ctx)
    end

    private def init_parameters
      @weights = []
      @biases = []
      @layer_dims.each_cons(2) do |dims|
        @weights << rnorm(dims)
        @biases << rnorm([dims[1]])
      end
      [*@weights, *@biases].each(&:attach_grad)
    end

    private def relu(x)
      ND.maximum(x, ND.zeros_like(x))
    end

    def forward(x)
      h = x
      n = @layer_dims.length
      (n - 2).times do |i|
        h_linear = ND.dot(h, @weights[i]) + @biases[i]
        h = relu(h_linear)
      end
      y_hat_linear = ND.dot(h, @weights[-1]) + @biases[-1]
    end

    private def softmax_cross_entropy(y_hat_linear, t)
      -ND.nansum(t * ND.log_softmax(y_hat_linear), axis: 0, exclude: true)
    end

    def loss(y_hat_linear, t)
      softmax_cross_entropy(y_hat_linear, t)
    end

    def predict(x)
      y_hat_linear = forward(x)
      ND.argmax(y_hat_linear, axis: 1)
    end
  end

  module_function

  def SGD(params, lr)
    params.each do |param|
      param[0..-1] = param - lr * param.grad
    end
  end

  def evaluate_accuracy(data_iter, model)
    num, den = 0.0, 0.0
    data_iter.each_with_index do |batch, i|
      data = batch.data[0].as_in_context(model.ctx)
      data = data.reshape([-1, model.layer_dims[0]])
      label = batch.label[0].as_in_context(model.ctx)

      predictions = model.predict(data)
      num += ND.sum(predictions == label)
      den += data.shape[0]
    end
    (num / den).as_scalar
  end

  def learning_loop(train_iter, test_iter, model,
                    epochs: 10, learning_rate: 0.001,
                    smoothing_constant: 0.01)
    epochs.times do |e|
      start = Time.now
      cumloss = 0.0
      num_batches = 0
      train_iter.each_with_index do |batch, i|
        data = batch.data[0].as_in_context(model.ctx)
        data = data.reshape([-1, model.layer_dims[0]])
        label = batch.label[0].as_in_context(model.ctx)
        label_one_hot = ND.one_hot(label, depth: model.layer_dims[-1])
        loss = MXNet::Autograd.record do
          y = model.forward(data)
          model.loss(y, label_one_hot)
        end
        loss.backward
        SGD(model.all_parameters, learning_rate)
        cumloss = ND.sum(loss).as_scalar
        num_batches += 1
      end
      test_acc = evaluate_accuracy(test_iter, model)
      train_acc = evaluate_accuracy(train_iter, model)
      duration = Time.now - start
      puts "Epoch #{e}. Loss: #{cumloss / (train_iter.batch_size * num_batches)}, " +
          "train-acc: #{train_acc}, test-acc: #{test_acc} (#{duration} sec)"
    end
  end
end
