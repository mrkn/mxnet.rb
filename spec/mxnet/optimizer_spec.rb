require 'spec_helper'

RSpec.describe MXNet::Optimizer do
  describe '.create' do
    it 'does not accept a class' do
      expect {
        MXNet::Optimizer.create(MXNet::Optimizer::SGD)
      }.to raise_error(ArgumentError)
    end
    it 'accepts an instance' do
      opt = MXNet::Optimizer.create(MXNet::Optimizer::SGD.new)
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
    it 'accepts a string' do
      opt = MXNet::Optimizer.create('sgd')
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
    it 'accepts a symbol' do
      opt = MXNet::Optimizer.create(:sgd)
      expect(opt).to be_a(MXNet::Optimizer::SGD)
    end
  end
end

RSpec.describe MXNet::Optimizer::Base do
  describe '#learning_rate' do
    it 'is changed by #learning_rate=' do
      o1 = MXNet::Optimizer::Base.new(learning_rate: 0.01)
      o1.learning_rate = 0.2
      expect(o1.learning_rate).to eq(0.2)
    end

    it 'is affected by lr_scheduler' do
      lr_s = MXNet::LRScheduler::FactorScheduler.new(step: 1)
      o2 = MXNet::Optimizer::Base.new(lr_scheduler: lr_s, learning_rate: 0.3)
      expect(o2.learning_rate).to eq(0.3)
      o2.lr_scheduler.base_lr = 0.4
      expect(o2.learning_rate).to eq(0.4)
    end
  end

  describe '#learning_rate=' do
    it 'raises RuntimeError if lr_scheduler is given' do
      lr_s = MXNet::LRScheduler::FactorScheduler.new(step: 1)
      o = MXNet::Optimizer::Base.new(lr_scheduler: lr_s, learning_rate: 0.3)
      expect { o.learning_rate = 0.5 }.to raise_error(RuntimeError)
    end
  end

  describe '#lr_mult' do
    specify do
      o = MXNet::Optimizer::Base.new
    end
  end

  describe '#lr_mult=' do
    specify do
      o = MXNet::Optimizer::Base.new()
    end
  end

  describe '#wd_mult=' do
  end

  describe 'lr_mult and wd_mult' do
    xspecify do
      data = MXNet::Symbol.var(:data)
      bias = MXNet::Symbol.var(:fc1_bias, lr_mult: 1.0)
      fc1 = MXNet::Symbol.FullyConnected(data: data, bias: bias, name: :fc1, num_hidden: 10, lr_mult: 0)
      fc2 = MXNet::Symbol.FullyConnected(data: fc1, name: :fc2, num_hidden: 10, wd_mult: 0.5)

      mod = MXNet::Module.new(symbol: fc2, label_names: nil, context: MXNet.default_context)
      mod.bind(data_shapes: [[:data, [5, 10]]])
      mod.init_params(init: MXNet::Init::Uniform.new(1.0))
      mod.init_optimizer(optimizer_params: {learning_rate: 1.0})
      args1, _ = mod.get_params
      args1.transform_values!(&:to_narray)

      mod.forward(MXNet::IO::DataBatch.new(data: [MXNet::Random::Uniform.new(low: -1.0, high: 1.0, shape: [5, 10])], label: nil), is_train: true)
      mod.backward(mod.get_outputs)
      mod.update
      args2, _ = mod.get_params
      args2.transform_values!(&:to_narray)

      expect(mod.optimizer.lr_mult).to include(fc1_bias: 1.0, fc1_weight: 0.0)
      expect(mod.optimizer.wd_mult).to include(fc2_bias: 0.5, fc2_weight: 0.5, fc1_bias: 0.0)
      expect((args1[:fc1_weight] - args2[:fc1_weight]).abs.max).to be <= 1e-10
      expect((args1[:fc1_bias] - args2[:fc1_bias]).abs.max).not_to be <= 1e-1
      expect((args1[:fc2_weight] - args2[:fc2_weight]).abs.max).not_to be <= 1e-1
    end
  end
end

RSpec.describe MXNet::Optimizer::SGD do
  context 'with lr_scheduler' do
    specify do
      opt = MXNet::Optimizer::SGD.new(
        momentum: 0.9,
        learning_rate: 0.1,
        lr_scheduler: MXNet::LRScheduler::FactorScheduler.new(step: 10, factor: 0.2)
      )
      expect(opt.learning_rate).to eq(0.1)
      expect { opt.learning_rate = 0.02 }.to raise_error
    end
  end

  describe '#update' do
    let(:optimizer) do
      MXNet::Optimizer::SGD.new(learning_rate: 0.1)
    end

    it 'updates the weight' do
      weight = MXNet::NDArray.array([1])
      gradient = MXNet::NDArray.array([0.5])
      optimizer.update(0, weight, gradient, nil)
      expect(weight.as_scalar).to be_within(0.01).of(0.95)
    end
  end
end

RSpec.xdescribe MXNet::Optimizer::Adam do

  describe '#create_state' do
  
  end

  describe '#update' do
  
  end

end

RSpec.xdescribe MXNet::Optimizer::Signum do
  describe '#update' do
  
  end
  describe '#create_state' do
  
  end
end


RSpec.xdescribe 'Sparse SGD' # TODO:
RSpec.xdescribe 'FTML' # TODO:
RSpec.xdescribe 'RMSProp' # TODO:
RSpec.xdescribe 'Ftrl' # TODO:
RSpec.xdescribe 'NADAM' # TODO:
