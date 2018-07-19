require 'spec_helper'
require 'mxnet/gluon'

RSpec.describe MXNet::Gluon::Parameter do
  specify do
    param = MXNet::Gluon::Parameter.new(:param)
    expect(param.name).to eq(:param)
    expect(param.dtype).to eq(:float32)
    expect(param.lr_mult).to eq(1.0)
    expect(param.wd_mult).to eq(1.0)
    expect(param.initializer).to eq(nil)
    expect(param.grad_req).to eq(:write)
    expect(param.shape).to eq(nil)
    expect { param.data }.to raise_error(RuntimeError, /'param' has not been initialized/)
    expect { param.list_data }.to raise_error(RuntimeError, /'param' has not been initialized/)
    expect { param.grad }.to raise_error(RuntimeError, /'param' has not been initialized/)
    expect { param.list_grad }.to raise_error(RuntimeError, /'param' has not been initialized/)
    expect { param.list_ctx }.to raise_error(RuntimeError, /'param' has not been initialized/)
    expect(param.zero_grad).to eq(nil)
  end

  describe '.new' do
    specify do
      expect(MXNet::Gluon::Parameter.new('test', shape: 1).shape).to eq([1])
      expect(MXNet::Gluon::Parameter.new('test', shape: [1]).shape).to eq([1])
    end
  end

  describe '#init' do
    specify do
      param = MXNet::Gluon::Parameter.new(:param)
      expect { param.init }.to raise_error(RuntimeError, /invalid shape/)
      param.shape = [2, 3]
      param.init
      expect(param.data).to be_a(MXNet::NDArray)
      expect(param.data.shape).to eq([2, 3])
      expect(param.data.dtype).to eq(:float32)

      old_data = param.data

      param = MXNet::Gluon::Parameter.new(:param)
      param.shape = [2, 3]
      param.dtype = :float64
      param.init
      param.init
      expect(param.data).not_to equal(old_data)
      expect(param.data.dtype).to eq(:float64)
    end

    specify 'force_reinit: true' do
      param = MXNet::Gluon::Parameter.new(:param)
      param.shape = [2, 3]
      param.init
      expect(param.data.dtype).to eq(:float32)

      old_data = param.data
      param.dtype = :float64
      param.init(force_reinit: true)
      expect(param.data).not_to equal(old_data)
      expect(param.data.dtype).to eq(:float64)
    end

    specify do
      param = MXNet::Gluon::Parameter.new(:param, shape: [2, 3])
      expect { param.init }.not_to raise_error(RuntimeError, /invalid shape/)
    end

    specify 'allow_deferred_init: true' do
      param = MXNet::Gluon::Parameter.new(:param, allow_deferred_init: true)
      expect { param.init }.not_to raise_error
    end

    specify do
      param = MXNet::Gluon::Parameter.new(:weight, shape: [10, 10])
      param.init(initializer: :xavier, ctx: [MXNet.cpu(0), MXNet.cpu(1)])
      expect(param.list_data.length).to eq(2)
      expect(param.list_grad.length).to eq(2)
      expect(param.list_ctx).to eq([MXNet.cpu(0), MXNet.cpu(1)])
      expect(param.data(ctx: MXNet.cpu(1)).context).to eq(MXNet.cpu(1))
      expect(param.data(ctx: MXNet.cpu(0)).shape).to eq([10, 10])
      expect(param.var.name).to eq(:weight)
    end

    context 'without deferred initialization' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', shape: [1]).tap do |parameter|
          parameter.init
        end
      end
      it 'initializes the data array' do
        expect(parameter.data).to be_a(MXNet::NDArray)
      end
      it 'initializes the grad array' do
        expect(parameter.grad).to be_a(MXNet::NDArray)
      end
      it 'attaches grads' do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          expect(p.grad).to eq(g)
        end
      end
    end
    context 'with deferred initialization' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', allow_deferred_init: true).tap do |parameter|
          parameter.init
          parameter.shape = [1]
          parameter.send(:_finish_deferred_init)
        end
      end
      it 'initializes the data array' do
        expect(parameter.data).to be_a(MXNet::NDArray)
      end
      it 'initializes the grad array' do
        expect(parameter.grad).to be_a(MXNet::NDArray)
      end
      it 'attaches grads' do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          expect(p.grad).to eq(g)
        end
      end
    end
  end
  describe '#list_ctx' do
    context 'without deferred initialization' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', shape: [1, 2]).tap do |parameter|
          parameter.init
        end
      end
      it 'returns the contexts' do
        expect(parameter.list_ctx).to eq([MXNet.cpu])
      end
    end
    context 'with deferred initialization' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', allow_deferred_init: true).tap do |parameter|
          parameter.init
        end
      end
      it 'returns the contexts' do
        expect(parameter.list_ctx).to eq([MXNet.cpu])
      end
    end
  end

  describe '#data' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('foo', shape: [1])
    end

    it 'fails if the parameter has not been initialized' do
      expect{parameter.data}.to raise_error(RuntimeError)
    end

    it 'fails if the parameter has not been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.data(ctx: MXNet.gpu)}.to raise_error(RuntimeError)
    end

    it 'succeeds if the parameter has been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.data(ctx: MXNet.cpu)}.not_to raise_error
    end

    it 'returns the initialized data' do
      parameter.init
      expect(parameter.data).to be_a(MXNet::NDArray)
    end
  end

  describe '#zero_grad' do
    # TODO:
    xspecify do
      param = MXNet::Gluon::Parameter.new(:param)
    end
  end

  describe '#grad' do
    let(:parameter) do
      MXNet::Gluon::Parameter.new('foo', shape: [1])
    end
    it 'fails if the parameter has not been initialized' do
      expect{parameter.grad}.to raise_error(RuntimeError)
    end
    it 'fails if the parameter has not been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.grad(ctx: MXNet.gpu)}.to raise_error(RuntimeError)
    end
    it 'succeeds if the parameter has been initialized on the specified context' do
      parameter.init(ctx: MXNet.cpu)
      expect{parameter.grad(ctx: MXNet.cpu)}.not_to raise_error
    end
    it 'returns the initialized grad' do
      parameter.init
      expect(parameter.grad).to be_a(MXNet::NDArray)
    end
  end
  describe '#shape=' do
    context 'with no shape' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo')
      end
      it 'assigns the shape' do
        parameter.shape = [1, 2]
        expect(parameter.shape).to eq([1, 2])
      end
    end
    context 'with incomplete shape' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', shape: [1, 0, 3])
      end
      it 'completes the shape' do
        parameter.shape = [1, 2, 3]
        expect(parameter.shape).to eq([1, 2, 3])
      end
    end
    context 'with shape' do
      let(:parameter) do
        MXNet::Gluon::Parameter.new('foo', shape: [1, 2])
      end
      it 'raises an error' do
        expect{parameter.shape = [1, 3]}.to raise_error(ArgumentError, /incompatible/)
      end
    end
  end

  describe '#var' do
    specify do
      param = MXNet::Gluon::Parameter.new(:param)
      expect(param.var).to be_a(MXNet::Symbol)
      expect(param.var).to equal(param.var)

      param = MXNet::Gluon::Parameter.new(:param, shape: [2, 3], dtype: :float64)
      expect(param.var.attr(:__shape__)).to eq([2, 3].to_s)
      expect(param.var.attr(:__dtype__)).to eq(MXNet::DType.name2id(:float64).to_s)
    end
  end

  describe '#reset_ctx' do
    specify do
      param = MXNet::Gluon::Parameter.new(:weight, shape: [10, 10])
      param.init(initializer: :xavier, ctx: [MXNet.cpu(0), MXNet.cpu(1)])
      expect(param.list_ctx).to eq([MXNet.cpu(0), MXNet.cpu(1)])
      param.reset_ctx([MXNet.cpu(1), MXNet.cpu(2)])
      expect(param.list_ctx).to eq([MXNet.cpu(1), MXNet.cpu(2)])
    end
  end
end

RSpec.describe MXNet::Gluon::Constant do
  specify do
    const = MXNet::Gluon::Constant.new(:const, [[1, 2], [3, 4]])
    const.init
    expect(const.name).to eq(:const)
    expect(const.data).to eq(MXNet::NDArray.arange(1, 5).reshape([2, 2]))
  end
end

RSpec.describe MXNet::Gluon::ParameterDict do
  let(:parent_params) do
    MXNet::Gluon::ParameterDict.new('foo_')
  end

  subject(:params) do
    MXNet::Gluon::ParameterDict.new('bar_', shared: parent_params)
  end

  specify do
    params = MXNet::Gluon::ParameterDict.new('net_')
    params.get('weight', shape: [10, 10])
    expect(params.keys).to eq(['net_weight'])
    params.init(ctx: MXNet.cpu)
    params.save('test.params')
    params.load('test.params', MXNet.cpu)
  end

  describe '#get' do
    specify do
      x = params.get(:x, shape: [2, 3], dtype: :float64)
      expect(x).to be_a(MXNet::Gluon::Parameter)
      expect(x.name).to eq('bar_x')
      expect(x.shape).to eq([2, 3])
      expect(x.dtype).to eq(:float64)

      expect(params.get(:x)).to equal(x)

      expect { params.get(:x, dtype: :float32) }.to raise_error(ArgumentError)

      expect(params.get(:x, foo: 42)[:foo]).to eq(42)
      expect { params.get(:x, foo: 1) }.to raise_error(ArgumentError)
    end

    context 'without a shared dict' do
      let(:parameter_dict) do
        MXNet::Gluon::ParameterDict.new
      end

      it 'creates a new parameter if not in dict' do
        expect(parameter_dict.get('foo')).to be_a(MXNet::Gluon::Parameter)
      end

      it 'retrieves a previously created parameter' do
        expect(parameter_dict.get('bar')).to equal(parameter_dict.get('bar'))
      end

      it 'uses keyword arguments to create a parameter' do
        expect(parameter_dict.get('baz', shape: [1, 1]).shape).to eq([1, 1])
      end
    end

    context 'with a shared dict' do
      let(:shared_dict) do
        MXNet::Gluon::ParameterDict.new.tap do |shared_dict|
          shared_dict.get('foo')
        end
      end

      let(:parameter_dict) do
        MXNet::Gluon::ParameterDict.new(shared: shared_dict).tap do |parameter_dict|
          parameter_dict.get('bar')
        end
      end

      it 'retrieves a parameter from the shared dict' do
        expect(parameter_dict.get('foo')).to equal(shared_dict.get('foo'))
      end
    end
  end

  describe '#get_constant' do
    context 'without initial value' do
      specify do
        expect { params.get_constant(:x) }.to raise_error(ArgumentError)
      end
    end

    context 'with initial value' do
      specify do
        x = params.get_constant(:x, [[1, 2], [3, 4]])
        x.init
        expect(x).to be_a(MXNet::Gluon::Constant)
        expect(x.shape).to eq([2, 2])
        expect(x.dtype).to eq(:float32)
        expect(x.data).to eq(MXNet::NDArray.array([[1, 2], [3, 4]]))

        expect(params.get_constant(:x)).to equal(x)

        expect {
          params.get_constant(:x, Numo::SFloat[1, 2, 3, 4])
        }.to raise_error(ArgumentError)

        params.get(:y)
        expect {
          params.get(:y, Numo::SFloat[1, 2, 3, 4])
        }.to raise_error(ArgumentError)
        # TODO: I think the following behavior is also collect,
        #       but Python's implementation doesn't behave as follow.
        # expect { params.get_constant(:y) }.to raise_error(ArgumentError)
      end
    end
  end

  describe '#update' do
    specify do
      params.get(:a)
      params.get(:b)
      other_params = MXNet::Gluon::ParameterDict.new('baz_')
      other_params.get(:x)
      other_params.get(:y)
      params.update(other_params)
      expect(params.keys).to contain_exactly('bar_a', 'bar_b', 'baz_x', 'baz_y')
    end

    specify do
      params.get(:a)
      params.get(:b)
      other_params = MXNet::Gluon::ParameterDict.new('bar_')
      other_params.get(:x)
      other_params.get(:b)
      expect { params.update(other_params) }.to raise_error(ArgumentError)
      expect(params.keys).to contain_exactly('bar_a', 'bar_b')
    end

    it 'copies parameters into dict' do
      other_params = MXNet::Gluon::ParameterDict.new('bar_')
      other_params.get('foo')
      params.update(other_params)
      expect(params.get('foo')).to equal(other_params.get('foo'))
    end

    it 'fails if parameters already exist' do
      other_params = MXNet::Gluon::ParameterDict.new('bar_')
      params.get('foo')
      other_params.get('foo')
      expect { params.update(other_params) }.to raise_error(ArgumentError)
    end
  end

  describe '#init' do
    specify do
      x = params.get(:x, shape: [2, 3])
      expect(x).to receive(:init).and_call_original

      expect { x.data }.to raise_error(RuntimeError)
      params.init
      expect(params.get(:x).data).to be_a(MXNet::NDArray)
      expect(x.data).to be_a(MXNet::NDArray)
      expect(x.data.shape).to eq([2, 3])
    end

    context 'with verbose: true' do
      specify do
        x = params.get(:x, shape: [2, 3])
        initializer = MXNet::Init::Uniform.new
        expect(initializer).to receive(:set_verbosity).with(true).and_call_original

        expect { x.data }.to raise_error(RuntimeError)
        params.init(initializer: initializer, verbose: true)
        expect(x.data).to be_a(MXNet::NDArray)
        expect(x.data.shape).to eq([2, 3])
      end
    end
  end

  describe '#zero_grad' do
    specify do
      x = params.get(:x, shape: [2, 3])
      expect(x).to receive(:zero_grad)
      params.init
      params.zero_grad
    end
  end

  describe '#reset_ctx' do
    specify do
      x = params.get(:x, shape: [2, 3])
      expect(x).to receive(:reset_ctx).with(MXNet.cpu(1))
      params.init
      params.reset_ctx(MXNet.cpu(1))
    end
  end

  describe '#setattr' do
    specify do
      x = params.get(:x, shape: [2, 3])
      y = params.get(:y, shape: [2, 3])
      params.set_attr(:z, MXNet::NDArray.array([[1, 2], [3, 4]]))
      expect(x[:z].to_narray).to eq(Numo::SFloat[[1, 2], [3, 4]])
      expect(y[:z].to_narray).to eq(Numo::SFloat[[1, 2], [3, 4]])
    end
  end

  describe '#save and #load' do
    specify do
      x = params.get(:x, shape: [2, 3])
      y = params.get(:y, shape: [3, 4])
      params.init
      params.save("test_params")

      other_params = MXNet::Gluon::ParameterDict.new('bar_')
      x = other_params.get(:x, shape: [2, 3])
      y = other_params.get(:y, shape: [3, 4])
      other_params.load("test_params", MXNet.cpu)
      expect(other_params.keys).to eq(['bar_x', 'bar_y'])
      expect(other_params.get(:x).data).to eq(x.data)
      expect(other_params.get(:y).data).to eq(y.data)
    end
  end
end
