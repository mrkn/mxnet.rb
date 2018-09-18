require 'spec_helper'

RSpec.describe MXNet::Autograd do
  describe '#record' do
    specify do
      expect { |b| MXNet::Autograd.record(&b) }.to yield_control
    end
  end

  describe '.mark_variables' do
    context 'first argument' do
      it 'must be Array' do
        expect{MXNet::Autograd.mark_variables(0, [])}.to raise_error(ArgumentError)
      end
    end

    context 'second argument' do
      it 'must be Array' do
        expect{MXNet::Autograd.mark_variables([], 0)}.to raise_error(ArgumentError)
      end
    end

    context 'first and second arguments' do
      it 'must be the same length' do
        args = [[1, 2], [1, 2, 3]]
        expect{MXNet::Autograd.mark_variables(*args)}.to raise_error(ArgumentError)
        args = [[], []]
        expect{MXNet::Autograd.mark_variables(*args)}.not_to raise_error
      end
    end

    context 'optional argument' do
      let(:zeros) do
        MXNet::NDArray.zeros(1)
      end

      it 'must be Symbol or Array' do
        args = [[], [], {grad_reqs: 'foo'}]
        expect{MXNet::Autograd.mark_variables(*args)}.to raise_error(ArgumentError)
        args = [[], [], {grad_reqs: :null}]
        expect{MXNet::Autograd.mark_variables(*args)}.not_to raise_error
        args = [[], [], {grad_reqs: []}]
        expect{MXNet::Autograd.mark_variables(*args)}.not_to raise_error
      end

      it 'must hold either :null, :write or :add' do
        args = [[zeros], [zeros], {grad_reqs: :foo}]
        expect{MXNet::Autograd.mark_variables(*args)}.to raise_error(ArgumentError)
        args = [[zeros], [zeros], {grad_reqs: [:foo]}]
        expect{MXNet::Autograd.mark_variables(*args)}.to raise_error(ArgumentError)
        [:null, :write, :add].each do |opt|
          args = [[zeros], [zeros], {grad_reqs: opt}]
          expect{MXNet::Autograd.mark_variables(*args)}.not_to raise_error
          args = [[zeros], [zeros], {grad_reqs: [opt]}]
          expect{MXNet::Autograd.mark_variables(*args)}.not_to raise_error
        end
      end
    end

    it 'attaches grads to vars' do
      vars = [MXNet::NDArray.random_uniform(shape: 1)]
      grads = [MXNet::NDArray.random_uniform(shape: 1)]
      MXNet::Autograd.mark_variables(vars, grads)
      expect(vars.map(&:grad)).to eq(grads)
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
