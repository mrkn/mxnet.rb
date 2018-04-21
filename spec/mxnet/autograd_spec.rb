require 'spec_helper'

RSpec.describe MXNet::Autograd do
  describe '#record' do
    specify do
      expect { |b| MXNet::Autograd.record(&b) }.to yield_control
    end
  end

  describe '.backward' do
    let(:x) { MXNet::NDArray.array([1, 2, 3, 4, 5]) }
    let(:y) { MXNet::NDArray.array([10]) }

    let(:w) { MXNet::NDArray::Random.normal(shape: x.shape) }
    let(:b) { MXNet::NDArray::Random.uniform(shape: [1]) }

    let(:loss) do
      w.attach_grad
      b.attach_grad

      expect((w.grad - MXNet::NDArray.zeros_like(w)).abs.max.as_scalar).to eq(0.0)
      expect((b.grad - MXNet::NDArray.zeros_like(b)).abs.max.as_scalar).to eq(0.0)

      MXNet::Autograd.record do
        z = MXNet::NDArray.dot(w, x) + b
        (y - z)**2 / 2.0
      end
    end

    specify do
      MXNet::Autograd.backward(loss)
      expect((w.grad - MXNet::NDArray.zeros_like(w)).abs.max.as_scalar).not_to eq(0.0)
      expect((b.grad - MXNet::NDArray.zeros_like(b)).abs.max.as_scalar).not_to eq(0.0)
    end

    specify do
      MXNet::Autograd.backward([loss])
      expect((w.grad - MXNet::NDArray.zeros_like(w)).abs.max.as_scalar).not_to eq(0.0)
      expect((b.grad - MXNet::NDArray.zeros_like(b)).abs.max.as_scalar).not_to eq(0.0)
    end
  end
end
