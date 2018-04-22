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

  def initialize(name: '')
    @name = name.to_s
  end

  attr_reader :name
  attr_reader :all_parameters
end

class Dense < BlockBase
  def initialize(in_units, out_units, ctx: MXNet::Context.default, name: 'dense')
    super(name: name)
    @in_units = in_units
    @out_units = out_units
    @ctx = ctx

    @weight = ND.random_normal(shape: [out_units, in_units], ctx: @ctx)
    @bias = ND.zeros([out_units], ctx: @ctx)

    @all_parameters = [@weight, @bias]
  end

  def forward(x)
    y = ND.FullyConnected(x, @weight, @bias, num_hidden: @out_units)
    return y
  end
end

class Conv2D < BlockBase
  def initialize(in_channels, num_filter, kernel:, stride:, pad:,
                 ctx: MXNet::Context.default, name: 'conv')
    super(name: name)
    @in_channels = in_channels
    @num_filter = num_filter
    @ctx = ctx

    @kernel = kernel.is_a?(Numeric) ? [kernel, kernel] : kernel.to_ary
    @stride = stride.is_a?(Numeric) ? [stride, stride] : stride.to_ary
    @pad    = pad.is_a?(Numeric) ? [pad, pad] : pad.to_ary

    @weight_shape, * = MXNet::Symbol.Convolution(
      MXNet::Symbol.var(:data, shape: [1, @in_channels, 0, 0]),
      kernel: @kernel, stride: @stride, pad: @pad,
      dilate: [1, 1], num_filter: @num_filter
    ).infer_shape_partial

    @weight = ND.random_normal(shape: @weight_shape[1], ctx: @ctx)
    @bias   = ND.zeros(@weight_shape[2], ctx: @ctx)

    @all_parameters = [@weight, @bias]
  end

  def forward(x)
    ND.Convolution(x, @weight, @bias, kernel: @kernel, stride: @stride,
                   dilate: [1, 1], pad: @pad, num_filter: @num_filter)
  end
end

class BatchNorm < BlockBase
  def initialize(n, eps: 1e-5, momentum: 0.9, ctx: MXNet::Context.default, name: 'bn')
    super(name: name)

    @n = n
    @eps = eps
    @momentum = momentum
    @ctx = ctx

    @running_mean = ND.zeros(n, ctx: @ctx)
    @running_var = ND.ones(n, ctx: @ctx)

    @gamma = ND.ones(n, ctx: @ctx)
    @beta = ND.zeros(n, ctx: @ctx)

    @all_parameters = [@gamma, @beta]
  end

  def forward(x)
    ND.BatchNorm(x, @gamma, @beta, @running_mean, @running_var,
                 eps: @eps, momentum: @momentum, fix_gamma: false)
  end
end

class AvgPool2D < BlockBase
  def initialize(kernel: [2, 2], stride: 1, pad: 0, ctx: MXNet::Context.default, name: 'avgpool')
    super(name: name)

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
  def initialize(in_channels, num_filter, stride: 1, ctx: MXNet::Context.default, name: 'resunit')
    super(name: name)
    @in_channels = in_channels
    @num_filter = num_filter
    @ctx = ctx

    @bn   = BatchNorm.new(in_channels, ctx: @ctx, name: "#{name}-bn")
    @conv = Conv2D.new(in_channels, num_filter, kernel: 3, stride: stride, pad: 1, ctx: @ctx, name: "#{name}-conv")
    if @in_channels != @num_filter
      @shortcut = Conv2D.new(in_channels, num_filter, kernel: 1, stride: stride, pad: 0, ctx: @ctx, name: "#{name}-conv-shortcut")
    end

    @all_parameters = [*@bn.all_parameters, *@conv.all_parameters]
    @all_parameters.concat @shortcut.all_parameters if @shortcut
  end

  def forward(x)
    bn  = @bn.forward(x)
    act = ND.Activation(bn, act_type: :relu)
    conv = @conv.forward(act)
    shortcut = @shortcut ? @shortcut.forward(x) : x
    return conv + shortcut
  end
end

class ResidualBlock < BlockBase
  def initialize(in_channels, num_filter, stride: 1, ctx: MXNet::Context.default, name: 'resblock')
    super(name: name)
    @unit1 = ResidualUnit.new(in_channels, num_filter, stride: stride, ctx: ctx, name: "#{name}-resunit1")
    @unit2 = ResidualUnit.new(num_filter, num_filter, stride: 1, ctx: ctx, name: "#{name}-resunit2")

    @all_parameters = [*@unit1.all_parameters, *@unit2.all_parameters]
  end

  def forward(x)
    unit1 = @unit1.forward(x)
    unit2 = @unit2.forward(unit1)
    return unit2
  end
end

class ResidualGroup < BlockBase
  def initialize(n, in_channels, num_filter, stride: 1, ctx: MXNet::Context.default, name: 'resgrp')
    super(name: name)
    @blocks = [ ResidualBlock.new(in_channels, num_filter, stride: stride, ctx: ctx, name: "#{name}-resblock1") ]
    1.step(n-1) do |i|
      @blocks << ResidualBlock.new(num_filter, num_filter, stride: 1, ctx: ctx, name: "#{name}-resblock#{i+1}")
    end

    @all_parameters = @blocks.map(&:all_parameters).flatten
  end

  def forward(x)
    @blocks.each do |block|
      x = block.forward(x)
    end
    return x
  end
end

class WideResNet < BlockBase
  ND = MXNet::NDArray

  def initialize(depth, width, in_channels, num_classes, ctx: MXNet::Context.default, name: '')
    unless (depth - 4) % 6 == 0
      raise ArgumentError, "depth must be 6n + 4"
    end
    @n = (depth - 4) / 6
    @widths = [16, *[16, 32, 64].map {|x| x * width }]
    @in_channels = in_channels
    @num_classes = num_classes
    @ctx = ctx

    build_network
  end

  private def build_network
    @conv1 = Conv2D.new(3, @widths[0], kernel: 3, stride: 1, pad: 1, ctx: @ctx, name: 'conv1')
    @groups = [
      ResidualGroup.new(@n, @widths[0], @widths[1], stride: 1, ctx: @ctx, name: 'resgrp1'),
      ResidualGroup.new(@n, @widths[1], @widths[2], stride: 2, ctx: @ctx, name: 'resgrp2'),
      ResidualGroup.new(@n, @widths[2], @widths[3], stride: 2, ctx: @ctx, name: 'resgrp3')
    ]
    @bn = BatchNorm.new(@widths[3], ctx: @ctx)
    @pool = AvgPool2D.new(kernel: [8, 8], stride: 1, pad: 0, ctx: @ctx)
    @fc = Dense.new(@widths[3], @num_classes, ctx: @ctx, name: 'fc')

    @all_parameters = [
      *@conv1.all_parameters,
      *@groups.map(&:all_parameters).flatten,
      *@bn.all_parameters,
      *@fc.all_parameters
    ]
  end

  def forward(x)
    x = @conv1.forward(x)
    @groups.each do |group|
      x = group.forward(x)
    end
    x = @bn.forward(x)
    x = ND.Activation(x, act_type: :relu)
    x = @pool.forward(x)
    x = x.reshape([0, -1]) # flatten
    x = @fc.forward(x)
    return x
  end
end

class MomentumSGD
  def initialize(lr: 0.01, lr_scheduler: nil, momentum: 0.0)
    @lr = lr
    @lr_scheduler = lr_scheduler
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
    if @states
      params.each_with_index do |param, i|
        MXNet::NDArray.sgd_mom_update(param, param.grad, @states[i], out: param,
                                      lr: lr, momentum: @momentum)
      end
    else
      params.each_with_index do |param, i|
        MXNet::NDArray.sgd_update(param, param.grad, out: param, lr: lr)
      end
    end
  end
end

def parse_options
  options = {
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
      '--log-interval=N', Integer, 'Number of batches to wait before logging.'
    ) {|v| options[:log_interval] = v }

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
def init_model(ctx: MXNet::Context.default)
  @model = WideResNet.new(28, 10, 3, 10, ctx: ctx)
  @model.all_parameters.each(&:attach_grad)
end

def test(ctx, val_iter)
  @metric.reset
  val_iter.reset
  val_iter.each do |batch|
    data = batch.data[0]
    label = batch.label[0]
    outputs = []

    outputs << @model.forward(data)

    @metric.update(label, outputs)
    break
  end
  return @metric.get
end

def train(opt, ctx)
end

def save_checkpoint(epoch, top1, best_acc, opt)
  return # TODO:
  if opt[:save_frequency] > 0 && (epoch + 1) % opt[:save_frequency] == 0
    fname = File.join(opt[:prefix], "wide_res_net_#{epoch}_acc_%.4f.params" % top1)
    @model.save_params(fname)
    puts "[Epoch #{epoch}] Saving checkpoint to #{fname} with Accuracy: %.4f" % top1
  end
  if top1 > best_acc[0]
    best_acc[0] = top1
    fname = File.join(opt[:prefix], "wide_res_net_best.params")
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
  init_model(ctx: context[0])

  # Loss function
  loss_func = SoftmaxCrossEntropyLoss.new

  # Initialize optimizer
  lr_scheduler = MXNet::LRScheduler::FactorScheduler.new(
    step: lr_decay_step, factor: lr_decay_factor)
  optimizer = MomentumSGD.new(
    lr: learning_rate,
    lr_scheduler: MXNet::LRScheduler::FactorScheduler.new(step: 80, factor: 0.2),
    momentum: momentum,
  )
  optimizer.init_states(@model.all_parameters)

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

      optimizer.update(@model.all_parameters)
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
