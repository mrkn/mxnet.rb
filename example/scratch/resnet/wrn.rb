require 'mxnet'
require_relative '../utils'

require 'optparse'

class SoftmaxCrossEntropyLoss
  def forward(pred, label)
    pred = MXNet::NDArray.log_softmax(pred, axis: -1)
    loss = -MXNet::NDArray.pick(pred, label, axis: -1, keepdims: true)
    return MXNet::NDArray.mean(loss, axis: 0, exclude: true)
  end
end

class BlockBase
  ND = MXNet::NDArray

  def initialize(prefix: nil, ctx: nil)
    @prefix = prefix || default_prefix
    @prefix = "#{@prefix}_" unless @prefix.empty?
    @name = @prefix.end_with?('_') ? @prefix[0...-1] : @prefix
    @context = ctx || MXNet::Context.default
    @children = {}
    @params = {}
  end

  private def default_prefix
    MXNet::Name::NameManager.current.get(nil, self.class.name.split('::').last.downcase)
  end

  attr_reader :prefix, :name, :context

  def [](name)
    @params[name] || @children[name]
  end

  def []=(name, val)
    case val
    when BlockBase
      register_child(val, name)
    when MXNet::NDArray
      @params[name] = val
    else
      raise ArgumentError, "invalid parameter type: #{val.class}"
    end
  end

  private def register_child(child, name=nil)
    name ||= @children.length.to_s
    @children[name] = child
  end

  def collect_params
    result = @params.values
    @children.each_value do |child|
      result.concat child.collect_params
    end
    return result
  end

  def collect_params_with_prefix(prefix=nil)
    prefix &&= "#{prefix}."
    result = @params.transform_keys {|k| :"#{prefix}#{k}" }
    @children.each do |child_name, child|
      result.update(child.collect_params_with_prefix("#{prefix}#{child_name}"))
    end
    return result
  end

  def save_params(filename)
    params = collect_params_with_prefix
    MXNet::NDArray.save(filename, params)
  end

  def load_params(filename)
    loaded = MXNet::NDArray.load(filename)
    return if loaded.nil? || loaded.empty?
    params = collect_params_with_prefix
    return if params.nil? || params.empty?

    loaded.each do |name, val|
      params[name.to_sym][0..-1] = loaded[name]
    end
  end
end

class Sequential < BlockBase
  def add(*blocks)
    blocks.each &method(:register_child)
  end

  def forward(x)
    @children.each_value do |child|
      x = child.forward(x)
    end
    return x
  end

  def length
    @children.length
  end
end

class Activation < BlockBase
  def initialize(act_type, **kwargs)
    super(**kwargs)
    @act_type = act_type
  end

  def forward(x)
    x = ND.Activation(x, act_type: @act_type)
    return x
  end
end

class Dense < BlockBase
  def initialize(in_units, out_units, use_bias: true, **kwargs)
    super(**kwargs)
    @in_units = in_units
    @out_units = out_units
    @use_bias = use_bias

    # weight is initialized by He Kaiming's method
    factor = @in_units
    scale = Math.sqrt(2.0 / factor)
    self[:weight] = ND.random_normal(loc: 0, scale: scale, shape: [out_units, in_units], ctx: context)
    self[:bias]   = ND.zeros([out_units], ctx: context) if @use_bias
  end

  def forward(x)
    y = ND.FullyConnected(x, self[:weight], @use_bias ? self[:bias] : nil,
                          no_bias: !@use_bias, num_hidden: @out_units)
    return y
  end
end

class Conv2D < BlockBase
  def initialize(in_channels, num_filter, kernel:, stride:, pad:, use_bias: true, **kwargs)
    super(**kwargs)
    @in_channels = in_channels
    @num_filter = num_filter
    @use_bias = use_bias

    @kernel = kernel.is_a?(Numeric) ? [kernel, kernel] : kernel.to_ary
    @stride = stride.is_a?(Numeric) ? [stride, stride] : stride.to_ary
    @pad    = pad.is_a?(Numeric) ? [pad, pad] : pad.to_ary

    @weight_shape, * = MXNet::Symbol.Convolution(
      MXNet::Symbol.var(:data, shape: [1, @in_channels, 0, 0]),
      kernel: @kernel, stride: @stride, pad: @pad,
      dilate: [1, 1], num_filter: @num_filter
    ).infer_shape_partial

    # weight is initialized by He Kaiming's method
    factor = @weight_shape[1][1]
    scale = Math.sqrt(2.0 / factor)
    self[:weight] = ND.random_normal(loc: 0, scale: scale, shape: @weight_shape[1], ctx: context)
    self[:bias]   = ND.zeros(@weight_shape[2], ctx: context) if @use_bias
  end

  def forward(x)
    ND.Convolution(x, self[:weight], @use_bias ? self[:bias] : nil,
                   no_bias: !@use_bias, kernel: @kernel, stride: @stride,
                   dilate: [1, 1], pad: @pad, num_filter: @num_filter)
  end
end

class BatchNorm < BlockBase
  def initialize(n, eps: 1e-5, momentum: 0.9, **kwargs)
    super(**kwargs)
    @n = n
    @eps = eps
    @momentum = momentum

    @running_mean = ND.zeros(n, ctx: context)
    @running_var = ND.ones(n, ctx: context)

    self[:gamma] = ND.ones(n, ctx: context)
    self[:beta]  = ND.zeros(n, ctx: context)
  end

  def forward(x)
    ND.BatchNorm(x, self[:gamma], self[:beta], @running_mean, @running_var,
                 eps: @eps, momentum: @momentum, fix_gamma: false)
  end
end

class AvgPool2D < BlockBase
  def initialize(kernel: [2, 2], stride: 1, pad: 0, **kwargs)
    super(**kwargs)

    @kernel = kernel.is_a?(Numeric) ? [kernel, kernel] : kernel.to_ary
    @stride = stride.is_a?(Numeric) ? [stride, stride] : stride.to_ary
    @pad = pad.is_a?(Numeric) ? [pad, pad] : pad.to_ary
  end

  def forward(x)
    ND.Pooling(x, kernel: @kernel, stride: @stride, pad: @pad,
               global_pool: false, pool_type: :avg,
               pooling_convention: :valid)
  end
end

class ResidualUnit < BlockBase
  def initialize(in_channels, num_filter, stride: 1, **kwargs)
    super(**kwargs)
    @in_channels = in_channels
    @num_filter = num_filter

    self[:body] = Sequential.new(prefix: '')
    self[:body].add(
      BatchNorm.new(in_channels),
      Activation.new(:relu),
      Conv2D.new(in_channels, num_filter, kernel: 3, stride: stride, pad: 1, use_bias: false)
    )

    if @in_channels != @num_filter
      self[:downsample] = Conv2D.new(in_channels, num_filter, kernel: 1, stride: stride, pad: 0, use_bias: false)
    end
  end

  def forward(x)
    residual = x
    x = self[:body].forward(x)
    residual = self[:downsample].forward(residual) if self[:downsample]
    return residual + x
  end
end

class ResidualBlock < BlockBase
  def initialize(in_channels, num_filter, stride: 1, **kwargs)
    super(**kwargs)
    self[:units] = Sequential.new(prefix: '')
    self[:units].add(
      ResidualUnit.new(in_channels, num_filter, stride: stride),
      ResidualUnit.new(num_filter, num_filter, stride: 1)
    )
  end

  def forward(x)
    return self[:units].forward(x)
  end
end

class ResidualGroup < BlockBase
  def initialize(n, in_channels, num_filter, stride: 1, **kwargs)
    super(**kwargs)
    self[:blocks] = Sequential.new(prefix: '')
    self[:blocks].add(ResidualBlock.new(in_channels, num_filter, stride: stride))
    1.step(n-1) do |i|
      self[:blocks].add(ResidualBlock.new(num_filter, num_filter, stride: 1))
    end
  end

  def forward(x)
    return self[:blocks].forward(x)
  end
end

class WideResNet < BlockBase
  ND = MXNet::NDArray

  def initialize(depth, width, in_channels, num_classes, **kwargs)
    super(**kwargs)
    unless (depth - 4) % 6 == 0
      raise ArgumentError, "depth must be 6n + 4"
    end
    @n = (depth - 4) / 6
    @widths = [16, *[16, 32, 64].map {|x| x * width }]
    @in_channels = in_channels
    @num_classes = num_classes

    build_network
  end

  private def build_network
    self[:features] = Sequential.new(prefix: '')
    self[:features].add(
      Conv2D.new(3, @widths[0], kernel: 3, stride: 1, pad: 1, use_bias: false),
      ResidualGroup.new(@n, @widths[0], @widths[1], stride: 1),
      ResidualGroup.new(@n, @widths[1], @widths[2], stride: 2),
      ResidualGroup.new(@n, @widths[2], @widths[3], stride: 2),
      BatchNorm.new(@widths[3]),
      AvgPool2D.new(kernel: [8, 8], stride: 1, pad: 0)
    )
    self[:output] = Dense.new(@widths[3], @num_classes)
  end

  def forward(x)
    x = self[:features].forward(x)
    # TODO: x = x.reshape([0, -1]) # flatten
    x = self[:output].forward(x)
    return x
  end
end

class NAG
  def initialize(lr: 0.01, lr_scheduler: nil, wd: 0.0, momentum: 0.0)
    @lr = lr
    @lr_scheduler = lr_scheduler
    @wd = wd
    @momentum = momentum
    @num_update = 0
    @states = nil
  end

  def learning_rate
    if @lr_scheduler
      @lr_scheduler.update(@num_update)
    else
      @lr
    end
  end

  def weight_decay
    @wd
  end

  def init_states(params)
    return unless @momentum > 0
    @states = params.map do |param|
      MXNet::NDArray.zeros(param.shape, ctx: param.context, dtype: param.dtype) # TODO: stype: param.stype
    end
    nil
  end

  def update(params)
    @num_update += 1
    lr = self.learning_rate
    wd = self.weight_decay

    @states.each_with_index do |state, i|
      param = params[i]
      grad = param.grad
      if state
        mom = state
        mom[0..-1].inplace * @momentum
        grad.inplace + wd * param
        mom[0..-1].inplace + grad
        grad[0..-1].inplace + @momentum * mom
        param[0..-1].inplace - lr * grad
      else
        raise "AssertionError: @momentum == 0.0" unless @momentum == 0.0
        param[0..-1].inplace - lr * (grad + wd * param)
      end
    end
  end
end

def parse_options
  options = {
    prefix: nil,
    data_dir: File.expand_path('../../../../spec/fixture/cifar10', __FILE__),
    batch_size: 128,
    context: [MXNet.cpu],
    learning_rate: 0.1,
    momentum: 0.9,
    weight_decay: 0.0001,
    lr_decay_factor: 0.8,
    lr_decay_step: 80,
    save_frequency: 10,
    start_epoch: 0,
    epochs: 300,
    log_interval: 50
  }
  OptionParser.new do |opt|
    opt.banner = "Usage: #{opt.program_name} [options] [DATA_DIR]"

    opt.on(
      '-b BATCH_SIZE', '--batch-size', Integer,
      'Specify learning batch size'
    ) {|v| options[:batch_size] = v }
    opt.on(
      '-g GPU_IDS', '--gpus', String,
      'Ordinates of GPUs to use, can be "0,1,2" or empty for cpu only.'
    ) do |gpus|
      gpus = gpus.strip.split(',')
      options[:context] = gpus.map {|g| MXNet.gpu(g.to_i) } unless gpus.empty?
    end

    opt.on(
      '--start-epoch=N', Integer, 'Starting epoch, 0 for fresh training, >0 to resume.'
    ) {|v| options[:start_epoch] = v }
    opt.on(
      '--resume=FILENAME', String, 'Path to saved weights where you want resume.'
    ) {|v| options[:resume] = v }
    opt.on(
      '--log-interval=N', Integer, 'Number of batches to wait before logging.'
    ) {|v| options[:log_interval] = v }
    opt.on(
      '--save-frequency=N', Integer, 'Number of epochs to save parameters.'
    ) {|v| options[:save_frequency] = v }

    opt.on('-h', '--help', 'Show help') do
      puts opt
      exit
    end

    opt.parse!(ARGV)

    num_gpus = options[:context].length
    options[:batch_size] *= num_gpus
  end
  return options
end

# Metrics
def init_metrics
  @metric = MXNet::Metric::CompositeEvalMetric.new([
    MXNet::Metric::Accuracy.new,
    MXNet::Metric::TopKAccuracy.new(top_k: 5)
  ])
end

# Model
def init_model(opt)
  MXNet::Context.with(opt[:context][0]) do
    @model = WideResNet.new(28, 10, 3, 10)
  end
  @model.load_params(opt[:resume]) if opt[:resume]

  @model_params = @model.collect_params
  @model_params.each(&:attach_grad)
end

def test(ctx, val_iter)
  @metric.reset
  val_iter.reset
  val_iter.each do |batch|
    data = batch.data[0].as_in_context(ctx[0])
    label = batch.label[0].as_in_context(ctx[0])
    outputs = []

    outputs << @model.forward(data)
    @metric.update(label, outputs)

    GC.start
  end
  return @metric.get
end

def train(opt, ctx)
end

def save_checkpoint(epoch, top1, best_acc, opt)
  if opt[:save_frequency] > 0 && (epoch + 1) % opt[:save_frequency] == 0
    fname = File.join(opt[:prefix] || Dir.pwd, "wide_res_net_#{epoch}_acc_%.4f.params" % top1)
    @model.save_params(fname)
    puts "[Epoch #{epoch}] Saving checkpoint to #{fname} with Accuracy: %.4f" % top1
  end
  if top1 > best_acc[0]
    best_acc[0] = top1
    fname = File.join(opt[:prefix] || Dir.pwd, "wide_res_net_best.params")
    @model.save_params(fname)
    puts "[Epoch #{epoch}] Saving checkpoint to #{fname} with Accuracy: %.4f" % top1
  end
end

def main
  opt = parse_options

  context = opt[:context]
  batch_size = opt[:batch_size]
  data_dir = opt[:data_dir]
  learning_rate = opt[:learning_rate]
  lr_decay_factor = opt[:lr_decay_factor]
  lr_decay_step = opt[:lr_decay_step]
  momentum = opt[:momentum]

  start_epoch = opt[:start_epoch]
  epochs = opt[:epochs]
  log_interval = opt[:log_interval]

  # Setup Data Iter
  Utils.get_cifar10(data_dir)
  train_iter, val_iter = Utils.get_cifar10_iter(
    batch_size: batch_size,
    data_dir: data_dir
  )

  init_metrics
  init_model(opt)

  # Loss function
  loss_func = SoftmaxCrossEntropyLoss.new

  # Initialize optimizer
  lr_scheduler = MXNet::LRScheduler::FactorScheduler.new(
    step: lr_decay_step, factor: lr_decay_factor)
  optimizer = NAG.new(
    lr: learning_rate,
    lr_scheduler: MXNet::LRScheduler::FactorScheduler.new(step: 80, factor: 0.2),
    wd: 0.0005,
    momentum: momentum,
  )
  optimizer.init_states(@model_params)

  # Training loop
  total_time = 0
  num_epochs = 0
  best_acc = [0]
  start_epoch.step(epochs) do |epoch|
    tic = Time.now
    train_iter.reset
    @metric.reset
    batch_tic = Time.now
    train_iter.each_with_index do |batch, i|
      data = batch.data[0].as_in_context(context[0])
      label = batch.label[0].as_in_context(context[0])

      # TODO: Collecting results from the different contexts
      outputs = []
      losses = []

      MXNet::Autograd.record do
        x, y = data, label # TODO: Support multiple contexts
        z = @model.forward(x)
        l = loss_func.forward(z, y)
        outputs << z
        losses << l
        MXNet::Autograd.backward(losses)
      end

      optimizer.update(@model_params)
      @metric.update(label, outputs)

      if log_interval && (i + 1) % log_interval == 0
        name, acc = @metric.get
        puts "Epoch #{epoch} Batch [#{i}]\tSpeed: " +
            "#{batch_size / (Time.now - batch_tic)} samples/sec\t" +
            "#{name[0]}=#{acc[0]}, #{name[1]}=#{acc[1]}"
      end
      batch_tic = Time.now

      GC.start
    end

    epoch_time = Time.now - tic

    # First epoch will usually be much slower than the subsequent epics,
    # so don't factor into the average
    total_time += epoch_time if num_epochs > 0
    num_epochs += 1

    name, acc = @metric.get
    puts "[Epoch #{epoch}] training: #{name[0]}=#{acc[0]}, #{name[1]}=#{acc[1]}"
    puts "[Epoch #{epoch}] time cost: #{epoch_time}"

    name, val_acc = test(context, val_iter)
    puts "[Epoch #{epoch}] validation: #{name[0]}=#{val_acc[0]}, #{name[1]}=#{val_acc[1]}"

    # save model if meet requirements
    save_checkpoint(epoch, val_acc[0], best_acc, opt)

    GC.start
  end

  if num_epochs > 1
    puts "Average epoch time: #{total_time / (num_epochs - 1)}"
  end
end

main if $0 == __FILE__
