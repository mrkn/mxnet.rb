require 'spec_helper'

module MXNet
  ::RSpec.describe Symbol do
    let(:nd_ones) { MXNet::NDArray.ones([2, 3]) }
    describe '#bind' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        ex = z.bind(MXNet.cpu, { x: nd_ones, y: nd_ones })
        ex.forward
        expect(ex.outputs[0].reshape([6]).to_a).to eq([2] * 6)
      end

      specify do
        x = MXNet.var(:x)
        expect { x.bind(MXNet.cpu, x: nd_ones) }.not_to raise_error
        expect { x.bind(MXNet.cpu, a: nd_ones) }.to raise_error(ArgumentError, /key `x` is missing in `args`/)
      end
    end

    describe '#simple_bind' do
      pending
    end

    describe '#eval' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones, y: nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([2] * 6)
      end
    end

    describe '#list_arguments' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        expect(x.list_arguments).to eq([:x])
        expect(z.list_arguments).to eq([:x, :y])
      end
    end

    describe '#list_outputs' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        expect(x.list_outputs).to eq([:x])
        expect(z.list_outputs[0].to_s).to match(/\A_plus\d+_output/)
      end
    end

    describe '#infer_shape' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        arg_shape, out_shape, = z.infer_shape(x: [2, 3], y: [2, 3])
        expect(arg_shape).to eq([[2, 3], [2, 3]])
        expect(out_shape).to eq([[2, 3]])

        expect { z.infer_shape(x: [2, 3], y: [2, 4]) }.to raise_error(MXNet::Error, /\AError in operator _plus\d+.+expected \(2,3\), got \(2,4\)/)
      end
    end

    describe '#infer_type' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        arg_type, out_type, = z.infer_type(x: :float32, y: :float32)
        expect(arg_type).to eq([:float32, :float32])
        expect(out_type).to eq([:float32])

        expect { z.infer_type(x: :int32, y: :float32) }.to raise_error(MXNet::Error, /\AError in operator _plus\d+.+expected int32, got float32/)
      end
    end

    describe '#save and .load' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        Dir.mktmpdir do |tmpdir|
          fn = File.join(tmpdir, 'symbol-z.json')
          z.save(fn)
          z2 = MXNet::Symbol.load(fn)
          expect(z2.to_json).to eq(z.to_json)
        end
      end
    end

    describe '#to_json and .load_json' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        Dir.mktmpdir do |tmpdir|
          z2 = MXNet::Symbol.load_json(z.to_json)
          expect(z2.to_json).to eq(z.to_json)
        end
      end
    end

    describe '#+' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x + y
        expect(z.name.to_s).to be_start_with('_plus')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones, y: nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([2.0] * 6)
      end
    end

    describe '#-' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x - y
        expect(z.name.to_s).to be_start_with('_minus')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones, y: nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([2.0] * 6)
      end
    end

    describe '#*' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x * y
        expect(z.name.to_s).to be_start_with('_mul')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones, y: nd_ones + nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([6.0] * 6)
      end
    end

    describe '#/' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x / y
        expect(z.name.to_s).to be_start_with('_div')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones, y: nd_ones + nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([1.5] * 6)
      end
    end

    describe '#%' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x % y
        expect(z.name.to_s).to be_start_with('_mod')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones, y: nd_ones + nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([1.0] * 6)
      end
    end

    describe '#**' do
      specify do
        x = MXNet.var(:x)
        y = MXNet.var(:y)
        z = x ** y
        expect(z.name.to_s).to be_start_with('_power')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones, y: nd_ones + nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([9.0] * 6)
      end
    end

    describe '#+@' do
      pending
    end

    describe '#-@' do
      specify do
        x = MXNet.var(:x)
        z = -x
        expect(z.name.to_s).to be_begin_with('_mul')
        ex = z.eval(ctx: MXNet.cpu, x: nd_ones + nd_ones + nd_ones)
        expect(ex[0].reshape([6]).to_a).to eq([-3] * 6)
      end
    end
  end
end
