require_relative 'mlp_scratch'
require 'optparse'
require 'pathname'

options = {
  batch_size: 256,
  gpu_id: nil,
  data_dir: Dir.pwd
}
OptionParser.new do |opt|
  opt.banner = "Usage: #{opt.program_name} [options] [DATA_DIR]"

  opt.on('-b BATCH_SIZE', '--batch-size', Integer, 'Specify learning batch size') {|v| options[:batch_size] = v }
  opt.on('-g [GPU_ID]', '--gpu', Integer, 'Specify GPU device ID') {|v| options[:gpu_id] = v }

  opt.on('-h', '--help', 'Show help') do
    puts opt
    exit
  end

  opt.parse!(ARGV)

  if options[:batch_size] < 1
    $stderr.puts "BATCH_SIZE should be greater than or equal to 1."
    puts opt
    abort
  end

  options[:gpu_id] = 0 if options[:gpu_id] == true
  if options[:gpu_id] && options[:gpu_id] < 0
    $stderr.puts "GPU_ID should be greater than or equal to 0."
    puts opt
    abort
  end

  if ARGV[0] && File.directory?(ARGV[0])
    options[:data_dir] = ARGV[0]
  end

  unless Pathname(options[:data_dir]).join('train-images-idx3-ubyte').file?
    $stderr.puts "Unable to find MNIST data in #{options[:data_dir]}"
    puts opt
    abort
  end
end

Dir.chdir options[:data_dir] do
  train_iter = MXNet::IO::MNISTIter.new(
    batch_size: options[:batch_size],
    shuffle: true
  )
  test_iter = MXNet::IO::MNISTIter.new(
    image: 't10k-images-idx3-ubyte',
    label: 't10k-labels-idx1-ubyte',
    batch_size: options[:batch_size],
    shuffle: false
  )

  if options[:gpu_id]
    gpu_model = MLPScratch::MLP.new(ctx: MXNet.gpu(options[:gpu_id]))
    MLPScratch.learning_loop(train_iter, test_iter, gpu_model)
  else
    cpu_model = MLPScratch::MLP.new
    MLPScratch.learning_loop(train_iter, test_iter, cpu_model)
  end
end
