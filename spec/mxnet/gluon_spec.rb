require 'spec_helper'

require 'mxnet/gluon'

RSpec.describe MXNet::Gluon do
  describe 'parameter_sharing' do
    let(:namespace) { Module.new }

    before do
      class namespace::Net < MXNet::Gluon::Block
        def initialize(**kwargs)
          super
          with_name_scope do
            @dense0 = MXNet::Gluon::NN::Dense.new(5, in_units: 5)
            @dense1 = MXNet::Gluon::NN::Dense.new(5, in_units: 5)
          end
        end

        def forward(x)
          @dense1.call(@dense2.call(x))
        end
      end
    end

    specify do
      net1 = namespace::Net.new(prefix: 'net1_')
      net2 = namespace::Net.new(prefix: 'net2_', params: net1.collect_params)
      net1.collect_params.init
      net2.call(MXNet::NDArray.zeros([3, 5]))

      net1.save_params('net1.params')

      net3 = namespace::Net.new(prefix: 'net3_')
      net3.load_params('net1.params', MXNet.cpu)
    end
  end

  describe 'parameter_str' do
    let(:namespace) { Module.new }

    before do
      class namespace::Net < MXNet::Gluon::Block
        def initialize(**kwargs)
          super
          with_name_scope do
            self[:dense0] = MXNet::Gluon::NN::Dense.new(10, in_units: 5, use_bias: false)
          end
        end
      end
    end

    specify do
      net = namespace::Net.new(prefix: 'net1_')
      lines = net.collect_params.to_s.lines
      expect(lines[0]).to be_start_with('net1_ (')
      expect(lines[1]).to include('net1_dense0_weight')
      expect(lines[1]).to include('[10, 5]')
      expect(lines[1]).to include('float32')
      expect(lines[2]).to eq(')')
    end
  end

  describe 'collect_parameters' do
    specify do
      net = MXNet::Gluon::NN::HybridSequential.new(prefix: 'test_')
      net.with_name_scope do
        net << MXNet::Gluon::NN::Conv2D.new(10, 3)
        net << MXNet::Gluon::NN::Dense.new(10, activation: :relu)
      end
      expect(net.collect_params.keys).to eq(['test_conv0_weight', 'test_conv0_bias', 'test_dense0_weight', 'test_dense0_bias'])
      expect(net.collect_params('.*weight').keys).to eq(['test_conv0_weight', 'test_dense0_weight'])
      expect(net.collect_params('test_conv0_bias|test_dense0_bias').keys).to eq(['test_conv0_bias', 'test_dense0_bias'])
    end
  end

  describe 'basic' do
    specify do
      model = MXNet::Gluon::NN::Sequential.new
      model << MXNet::Gluon::NN::Dense.new(128, activation: :tanh, in_units: 10, flatten: false)
      model << MXNet::Gluon::NN::Dropout.new(0.5)
      model << MXNet::Gluon::NN::Dense.new(64, activation: :tanh, in_units: 256)
      model << MXNet::Gluon::NN::Dense.new(32, activation: :tanh, in_units: 64)
      model << MXNet::Gluon::NN::Activation.new(:relu)

      # symbol
      x = MXNet::Symbol.var(:data)
      y = model.call(x)
      expect(y.list_arguments.length).to eq(7)

      # ndarray
      model.collect_params.init(MX::Init:: Xavier.new(magnitude: 2.24))
      x = model.call(MXNet::NDArray.zeros([32, 2, 10]))
      expect(x.shape).to eq([32, 32])
      x.wait_to_read

      model.collect_params.setattr(:grad_req, 'null')
      expect(model.collect_params.values[0]._grad).to eq(nil)
      model.collect_params.setattr(:grad_req, 'write')
      expect(model.collect_params.values[0]._grad).not_to eq(nil)
    end
  end

  describe 'dense' do
    specify 'with flattgen: false' do
      model = MXNet::Gluon::NN::Dense.new(128, activation: :tanh, in_units: 10, flatten: false, prefix: 'test_')
      inputs = MXNet::Symbol.var(:data)
      outputs = model.call(inputs)
      expect(model.collect_params.keys).to contain_exactly('test_weight', 'test_bias')
      expect(output.list_outputs).to eq(['test_tanh_fwd_output'])
      args, outs, auxs = outputs.infer_shape(data: [2, 3, 10])
      expect(outs).to eq([[2, 3, 128]])
    end

    specify 'with flatten: true' do
      model = MXNet::Gluon::NN::Dense.new(128, activation: :relu, in_units: 30, flatten: true, prefix: 'test2_')
      inputs = MXNet::Symbol.var(:data)
      outputs = model.call(inputs)
      expect(model.collect_params.keys).to contain_exactly('test2_weight', 'test2_bias')
      expect(outputs.list_outputs).to eq(['test2_relu_fwd_output'])
      args, outs, auxs = outputs.infer_shape(data: [17, 2, 5, 3])
      expect(outs).to eq([[17, 128]])
    end
  end

  describe 'symbol_block' do
    let(:namespace) { Module.new }
    before do
      class namespace::Net < MXNet::Gluon::HybridBlock
        def initialize(model)
          super()
          @model = model
        end

        def hybrid_forward(mod, x)
          out = @model.call(x)
          mod.add_n(*out.map(&:sum))
        end
      end
    end

    specify do
      model = MXNet::Gluon::NN::HybridSequential.new
      model << MXNet::Gluon::NN::Dense.new(128, activation: :tanh)
      model << MXNet::Gluon::NN::Dropout.new(0.5)
      model << MXNet::Gluon::NN::Dense.new(64, activation: :tanh)
      model << MXNet::Gluon::NN::Dense.new(32, in_units: 64)
      model << MXNet::Gluon::NN::Activation.new(:relu)

      model.init

      inputs = MXNet::Symbol.var(:data)
      outputs = model.call(inputs).internals

      smodel = MXNet::Gluon::SymbolBlock.new(outputs, inputs, params: model.collect_params)

      expect(smodel.call(MXNet::NDArray.zeros([16, 10])).length).to eq(14)

      out = smodel.call(MXNet::Symbol.var(:in))
      expect(out.length).to eq(outputs.list_outputs.length)

      net = namespace::Net.new(smodel)
      net.hybridize
      expect(net.call(MXNet::NDArray.zeros([16, 10]))).to be_a(MXNet::NDArray)

      # the case without using `internals`
      inputs = MXNet::Symbol.var(:data)
      outputs = model.call(inputs)
      smodel = MXNet::Gluon::SymbolBlock.new(outputs, inputs, params: model.collect_params)
      net = namespace::Net.new(smodel)
      net.hybridize
      expect(net.call(MXNet::NDArray.zeros([16, 10]))).to be_a(MXNet::NDArray)
    end
  end

  def check_layer_forward(layer, dshape)
    layer.collect_params.init
    x = MXNet::NDArray.ones(shape: dshape)
    x.attach_grad
    out = MXNet::Autograd.record do
      layer.call(x)
    end
    out.backward

    nary_out = out.to_narray
    nary_dx  = x.grad.to_narray

    layer.hybridize

    x = MXNet::NDArray.ones(shape: dshape)
    x.attach_grad
    out = MXNet::Autograd.record do
      layer.call(x)
    end
    out.backward

    # assert_almost_equal(nary_out, out.to_narray, rtol: 1e-5, atol: 1e-6)
    expect((nary_out - out.to_narray).abs.max).to be < 1e-5
    # assert_almost_equal(nary_dx, x.grad.to_narray, rtol: 1e-5, atol: 1e-6)
    expect((nary_dx - x.grad.to_narray).abs.max).to be < 1e-5
  end

  describe 'conv' do
    specify do
      layers_1d = [
        MXNet::Gluon::NN::Conv1D.new(16, 3, in_channels: 4),
        MXNet::Gluon::NN::Conv1D.new(16, 3, groups: 2, in_channels: 4),
        MXNet::Gluon::NN::Conv1D.new(16, 3, strides: 3, groups: 2, in_channels: 4),
      ]
      layers_1d.each do |layer|
        check_layer_forward(layer, [1, 4, 10])
      end

      layers_2d = [
        MXNet::Gluon::NN::Conv2D.new(16, [3, 4], in_channels: 4),
        MXNet::Gluon::NN::Conv2D.new(16, [5, 4], in_channels: 4),
        MXNet::Gluon::NN::Conv2D.new(16, [3, 4], groups: 2, in_channels: 4),
        MXNet::Gluon::NN::Conv2D.new(16, [3, 4], strides: 4, in_channels: 4),
        MXNet::Gluon::NN::Conv2D.new(16, [3, 4], dilation: 4, in_channels: 4),
        MXNet::Gluon::NN::Conv2D.new(16, [3, 4], padding: 4, in_channels: 4),
      ]
      layers_2d.each do |layer|
        check_layer_forward(layer, [1, 4, 20, 20])
      end

      layers_3d = [
        MXNet::Gluon::NN::Conv3D.new(16, [1, 8, 4], in_channels: 4, activation: :relu),
        MXNet::Gluon::NN::Conv3D.new(16, [5, 4, 3], in_channels: 4),
        MXNet::Gluon::NN::Conv3D.new(16, [3, 3, 3], groups: 2, in_channels: 4),
        MXNet::Gluon::NN::Conv3D.new(16, 4, strides: 4, in_channels: 4),
        MXNet::Gluon::NN::Conv3D.new(16, [3, 3, 3], padding: 4, in_channels: 4),
      ]
      layers_3d.each do |layer|
        check_layer_forward(layer, [1, 4, 10, 10, 10])
      end

      layer = MXNet::Gluon::NN::Conv2D.new(16, [3, 3], layout: 'NHWC', in_channels: 4)
      # TODO: check_layer_forward(layer, [1, 10, 10, 4])

      layer = MXNet::Gluon::NN::Conv3D.new(16, [3, 3, 3], layout: 'NDHWC', in_channels: 4)
      # TODO: check_layer_forward(layer, [1, 10, 10, 10, 4])
    end
  end

  xdescribe 'deconv'

  describe 'pool' do
    specify do
      layers_1d = [
        MXNet::Gluon::NN::MaxPool1D.new,
        MXNet::Gluon::NN::MaxPool1D.new(3),
        MXNet::Gluon::NN::MaxPool1D.new(3, 2),
        MXNet::Gluon::NN::AvgPool1D.new,
        MXNet::Gluon::NN::GlobalAvgPool1D.new,
      ]
      layers_1d.each do |layer|
        check_layer_forward(layer, [1, 2, 10])
      end

      layers_2d = [
        MXNet::Gluon::NN::MaxPool2D.new,
        MXNet::Gluon::NN::MaxPool2D.new([3, 3]),
        MXNet::Gluon::NN::MaxPool2D.new(3, 2),
        MXNet::Gluon::NN::AvgPool2D.new,
        MXNet::Gluon::NN::GlobalAvgPool2D.new,
      ]
      layers_2d.each do |layer|
        check_layer_forward(layer, [1, 2, 10, 10])
      end

      layers_3d = [
        MXNet::Gluon::NN::MaxPool3D.new,
        MXNet::Gluon::NN::MaxPool3D.new([3, 3, 3]),
        MXNet::Gluon::NN::MaxPool3D.new(3, 2),
        MXNet::Gluon::NN::AvgPool3D.new,
        MXNet::Gluon::NN::GlobalAvgPool3D.new,
      ]
      layers_3d.each do |layer|
        check_layer_forward(layer, [1, 2, 10, 10, 10])
      end

      # test ceil_mode
      x = MXNet::NDArray.zeros([2, 2, 10, 10])

      layer = MXNet::Gluon::NN::MaxPool2D.new(3, ceil_mode: false)
      layer.collect_params.init
      expect(layer.call(x).shape).to eq([2, 2, 3, 3])

      layer = MXNet::Gluon::NN::MaxPool2D.new(3, ceil_mode: true)
      layer.collect_params.init
      expect(layer.call(x).shape).to eq([2, 2, 4, 4])
    end
  end

  describe 'batchnorm' do
    specify do
      layer = MXNet::Gluon::NN::BatchNorm.new(in_channels: 10)
      check_layer_forward(layer, [2, 10, 10, 10])
    end
  end

  describe 'reshape' do
    specify do
      x = MXNet::NDArray.ones([2, 4, 10, 10])
      layer = MXNet::Gluon::NN::Conv2D.new(10, 2, in_channels: 4)
      layer.collect_params.init
      x = MXNet::Autograd.record do
        x = layer.call(x)
        x = x.reshape([-1])
        x + 10
      end
      x.backward
    end
  end

  describe 'slice' do
    specify do
      x = MXNet::NDArray.ones([5, 4, 10, 10])
      layer = MXNet::Gluon::NN::Conv2D.new(10, 2, in_channels: 4)
      layer.collect_params.init
      x = MXNet::Autograd.record do
        x = layer.call(x)
        x = x[1...3]
        x + 10
      end
      x.backward
    end
  end

  describe 'at' do
    specify do
      x = MXNet::NDArray.ones([5, 4, 10, 10])
      layer = MXNet::Gluon::NN::Conv2D.new(10, 2, in_channels: 4)
      layer.collect_params.init
      x = MXNet::Autograd.record do
        x = layer.call(x)
        x = x[1]
        x + 10
      end
      x.backward
    end
  end

  describe 'deferred_init' do
    specify do
      x = MXNet::NDArray.ones([5, 4, 10, 10])
      layer = MXNet::Gluon::NN::Conv2D.new(10, 2)
      layer.collect_params.init
      layer.call(x)
    end
  end

  def check_split_data(x, num_slice, batch_axis, **kwargs)
    res = MXNet::Gluon::Utils.split_data(x, num_slice, batch_axis, **kwargs)
    expect(res.length).to eq(num_slice)
    # TODO: almost_equal(MXNet::NDArray.concat(*res, dim: batch_axis).to_narray, x.to_narray)
  end

  describe 'split_data' do
    specify do
      x = MXNet::NDArray::Random.uniform(shape: [128, 33, 64])
      check_split_data(x, 8, 0)
      check_split_data(x, 3, 1)
      check_split_data(x, 4, 1, even_split: false)
      check_split_data(x, 15, 1, even_split: false)
      expect { check_split_data(x, 4, 1) }.to raise_error(ArgumentError, /cannot be evenly split/)
    end
  end

  describe 'flatten' do
    specify do
      flatten = MXNet::Gluon::NN::Flatten.new
      x = MXNet::NDArray.zeros([3, 4, 5, 6])
      expect(flatten.call(x).shape).to eq([3, 4*5*6])
      x = MXNet::NDArray.zeros([3, 6])
      expect(flatten.call(x).shape).to eq([3, 6])
      x = MXNet::NDArray.zeros([3])
      expect(flatten.call(x).shape).to eq([3, 1])
    end
  end

  xdescribe 'trainer'

  describe 'block_attr_hidden' do
    specify 'regular attributes can change types' do
      b = MXNet::Gluon::Block.new
      b[:a] = nil
      b[:a] = 1
    end
  end

  xdescribe 'block_attr_block'

  xdescribe 'block_attr_param'

  xdescribe 'block_attr_regular'

  xdescribe 'block_attr_list_of_block'

  xdescribe 'sequential_warning'

  xdescribe 'global_norm_clip'

  xdescribe 'embedding' do
    specify do
      layer = MXNet::Gluon::NN::Embedding.new(10, 100)
      layer.init
      x = MXNet::NDArray.array([3, 4, 2, 0, 1]) # TODO: NDArray.array
      MXNet::Autograd.record do
        y = layer.call(x)
        y.backward
      end
      expect((layer.weight.grad[0...5].to_narray - 1).abs.max).to eq(0)
      expect(layer.weight.grad[5..-1].to_narray.abs.max).to eq(0)
    end
  end

  xdescribe 'export'

  xdescribe 'hybrid_stale_cache'

  describe 'lambda' do
    specify do
      net1 = MXNet::Gluon::NN::HybridSequential.new
      net1 << MXNet::Gluon::NN::Activation.new('tanh')
      net1 << MXNet::Gluon::NN::LeakyReLU.new(0.1)

      net2 = MXNet::Gluon::NN::HybridSequential.new
      net2 << MXNet::Gluon::NN::HybridLabmda.new('tanh')
      net2 << MXNet::Gluon::NN::HybridLabmda.new do |mod, x, *args|
        mod.LeakyReLU(x, *args, slope: 0.1)
      end

      net3 = MXNet::Gluon::NN::Sequential.new
      net3 << MXNet::Gluon::NN::Lambda.new('tanh')
      net3 << MXNet::Gluon::NN::Lambda.new do |x|
        MXNet::NDArray.LeakyReLU(x, slope: 0.1)
      end

      input_data = MXNet::NDArray::Random.uniform(shape: [2, 3, 5, 7])
      out1 = net1.call(input_data)
      out2 = net2.call(input_data)
      out3 = net3.call(input_data)
      expect((out1.to_narray - out2.to_narray).abs.maximum).to be <= 1e-7
      expect((out1.to_narray - out3.to_narray).abs.maximum).to be <= 1e-7
    end
  end

  describe 'fill_shape_deferred' do
    specify do
      net = MXNet::Gluon::NN::HybridSequential.new
      net.with_name_scope do
        net << MXNet::Gluon::NN::Conv2D.new(64, kernel_size: 2, padding: 1)
        net << MXNet::Gluon::NN::BatchNorm.new
        net << MXNet::Gluon::NN::Dense.new(10)
      end
      net.hybridize
      net.init
      net.call(MXNet::NDArray.ones([2,3,5,7], ctx: ctx))
      expect(net[0].weight.shape[1]).to eq(3)
      expect(net[1].gamma.shape[0]).to eq(64)
      expect(net[2].weight.shape[1]).to eq(3072)
    end
  end

  describe 'dtype' do
    specify do
      net = MXNet::Gluon::ModelZoo::Vision::ResNet18_v1.new
      net.init
      net.cast(:float64)
      MXNet::Autograd.record do
        y = net.call(MXNet::NDArray.ones([16, 3, 32, 32], dtype: :float64))
        y.backward
      end

      net = MXNet::Gluon::ModelZoo::Vision::ResNet18_v1
      net.init
      net.hybridize
      net.call(MXNet::NDArray.ones([16, 3, 32, 32], dtype: :float32))

      net.cast(:float64)
      net.call(MXNet::NDArray.ones([16, 3, 32, 32], dtype: :float64))

      MXNet::NDArray.wait_all
    end
  end

  describe 'fill_shape_load' do
    specify do
      ctx = MXNet::Context.current
      net1 = MXNet::Gluon::NN::HybridSequential.new
      net1.with_name_scope do
        net1 << MXNet::Gluon::NN::Conv2D.new(64, kernel_size: 2, padding: 1)
        net1 << MXNet::Gluon::NN::BatchNorm.new
        net1 << MXNet::Gluon::NN::Dense.new(10)
      end
      net1.hybridize
      net1.init(ctx: ctx)
      net1.call(MXNet::NDArray.ones([2,3,5,7], ctx: ctx))
      net1.save_params('net_fill.params')

      net2 = MXNet::Gluon::NN::HybridSequential.new
      net2.with_name_scope do
        net2 << MXNet::Gluon::NN::Conv2D.new(64, kernel_size: 2, padding: 1)
        net2 << MXNet::Gluon::NN::BatchNorm.new
        net2 << MXNet::Gluon::NN::Dense.new(10)
      end
      net2.hybridize
      net2.init
      net2.load_params('net_fill.params', ctx: ctx)

      expect(net2[0].weight.shape[1]).to eq(3)
      expect(net2[1].gamma.shape[0]).to eq(64)
      expect(net2[2].weight.shape[1]).to eq(3072)
    end
  end

  describe 'inline' do
    specify do
      net = MXNet::Gluon::NN::HybridSequential.new
      net.with_name_scope do
        net << MXNet::Gluon::NN::Dense.new(10)
        net << MXNet::Gluon::NN::Dense.new(10)
        net << MXNet::Gluon::NN::Dense.new(10)
      end

      net.init
      net.hybridize(inline_limit: 3)
      y = MXNet::Autograd.record do
        net.call(MXNet::NDArray.zeros([1, 10])) # TODO: rename to forward
      end

      len_1 = JSON.load(MXNet::Autograd.get_symbol(y).to_json)['nodes'].length
      y.backward

      net.hybridize(inline_limit: 0)
      y = MXNet::Autograd.record do
        net.call(MXNet::NDArray.zeros([1, 10]))
      end

      len_2 = JSON.load(MXNet::Autograd.get_symbol(y).to_json)['nodes'].length
      y.backward

      expect(len_1).to eq(len_2 + 2)
    end
  end
end
