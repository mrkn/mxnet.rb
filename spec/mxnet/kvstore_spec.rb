require 'spec_helper'

RSpec.describe MXNet::KVStore do
  describe '.new' do
    specify do
      kvs = MXNet::KVStore.new
      expect(kvs.type).to eq(:local)

      kvs = MXNet::KVStore.new(:local)
      expect(kvs.type).to eq(:local)

      kvs = MXNet::KVStore.new('local')
      expect(kvs.type).to eq(:local)

      kvs = MXNet::KVStore.new(:device)
      expect(kvs.type).to eq(:device)

      kvs = MXNet::KVStore.new('device')
      expect(kvs.type).to eq(:device)

      kvs = MXNet::KVStore.new(:dist_sync)
      expect(kvs.type).to eq(:dist_sync)

      kvs = MXNet::KVStore.new(:dist_device_sync)
      expect(kvs.type).to eq(:dist_device_sync)

      kvs = MXNet::KVStore.new(:dist_async)
      expect(kvs.type).to eq(:dist_async)

      expect {
        kvs = MXNet::KVStore.new(:invalid_kvstore_name)
      }.to raise_error(ArgumentError)
    end
  end

  subject(:kvs_local) { MXNet::KVStore.new(:local) }

  describe '#init' do
    specify do
      subject.init('3', MXNet::NDArray.ones([2, 3]) * 2)
      x = MXNet::NDArray.zeros([2, 3])
      subject.pull('3', out: x)
      expect(x.to_narray).to eq(Numo::DFloat[[2, 2, 2], [2, 2, 2]])
    end

    specify do
      keys = %w[5 7 9]
      subject.init(keys, Array.new(keys.length) { MXNet::NDArray.ones([2, 3]) })
      x = MXNet::NDArray.zeros([2, 3])
      subject.pull('5', out: x)
      expect(x.to_narray).to eq(Numo::DFloat[[1, 1, 1], [1, 1, 1]])
      subject.pull('7', out: x)
      expect(x.to_narray).to eq(Numo::DFloat[[1, 1, 1], [1, 1, 1]])
      subject.pull('9', out: x)
      expect(x.to_narray).to eq(Numo::DFloat[[1, 1, 1], [1, 1, 1]])
    end

    # TODO: sparse storage types support
    xspecify do
      subject.init('4', MXNet::NDArray.ones([2, 3]).to_stype(:row_sparse))
      b = MXNet::NDArray::Sparse.zeros(:row_sparse, [2, 3])
      suject.row_sparse_pull('4', row_ids: MXNet::NDArray.array([0, 1]), out: b)
      expect(b.to_narray).to eq(Numo::DFloat[[1, 1, 1], [1, 1, 1]])
    end
  end

  describe '#push' do
    specify do
      subject.push('3',)
    end
  end
end

RSpec.describe 'tests KVStore imported from Python' do
  let(:shape) { [4, 4] }
  let(:keys) { [5, 7, 11] }
  let(:str_keys) { ['b', 'c', 'd'] }

  def init_kv(stype=:default)
    kv = MXNet::KVStore.new
    # single
    kv.init(3, MXNet::NDArray.zeros(shape)) # TODO: stype
    # array
    kv.init(keys, [MXNet::NDArray.zeros(shape)]*keys.length) # TODO: stype
    return kv
  end

  def init_kv_with_str(stype=:default)
    kv = MXNet::KVStore.new
    # single
    kv.init('a', MXNet::NDArray.zeros(shape)) # TODO: stype
    # array
    kv.init(str_keys, [MXNet::NDArray.zeros(shape)] * keys.length) # TODO: stype
    return kv
  end

  def check_diff_to_scalar(x, y)
    expect((x - y).to_narray.sum).to be_within(1e-15).of(0)
  end

  describe 'single kv pair' do
    def check_single_kv_pair(kv, key)
      kv.push(key, MXNet::NDArray.ones(shape))
      val = MXNet::NDArray.empty(shape)
      kv.pull(key, out: val)
      check_diff_to_scalar(val, 1)
    end

    specify do
      check_single_kv_pair(init_kv, 3)
    end

    specify do
      check_single_kv_pair(init_kv_with_str, 'a')
    end
  end

  describe 'multiple kv pairs by array' do
    def check_list_kv_pair(kv, key)
      kv.push(key, [MXNet::NDArray.ones(shape)*4] * key.length)
      val = [MXNet::NDArray.empty(shape)] * key.length
      kv.pull(key, out: val)
      val.each do |v|
        check_diff_to_scalar(v, 4)
      end
    end

    specify do
      check_list_kv_pair(init_kv, keys)
    end

    specify do
      check_list_kv_pair(init_kv_with_str, str_keys)
    end
  end

  describe 'aggregation values on multiple devices' do
    def check_aggregator(kv, key, key_ary)
      # devices
      num_devs = 4
      devs = Array.new(num_devs) {|i| MXNet::Context.new(:cpu, i) }

      # single
      vals = devs.map {|d| MXNet::NDArray.ones(shape, ctx: d) }
      kv.push(key, vals)
      kv.pull(key, out: vals)
      vals.each do |v|
        check_diff_to_scalar(v, num_devs)
      end

      # multiple
      vals = key_ary.map { devs.map {|d| MXNet::NDArray.ones(shape, ctx: d) * 2.0 } }
      kv.push(key_ary, vals)
      kv.pull(key_ary, out: vals)
      vals.each do |vv|
        vv.each do |v|
          check_diff_to_scalar(v, num_devs * 2.0)
        end
      end
    end

    specify do
      check_aggregator(init_kv, 3, keys)
    end

    specify do
      check_aggregator(init_kv_with_str, 'a', str_keys)
    end
  end

  describe 'updater' do
    let(:updater) do
      ->(key, recv, local) do
        raise unless key.is_a? Integer
        local.inplace + recv
      end
    end

    let(:str_updater) do
      ->(key, recv, local) do
        raise unless key.is_a? String
        local.inplace + recv
      end
    end

    def check_updater(kv, key, key_ary)
      # devices
      num_devs = 4
      devs = Array.new(num_devs) {|i| MXNet::Context.new(:cpu, i) }

      # single
      vals = devs.map {|d| MXNet::NDArray.ones(shape, d) }
      kv.push(key, vals)
      kv.pull(key, out: vals)
      vals.each do |v|
        check_diff_to_scalar(v, num_devs)
      end

      # multiple
      vals = key_ary.map { devs.map {|d| MXNet::NDArray.ones(shape, d) } }
      num_push = 4
      num_push.times do |i|
        kv.push(key_ary, vals)
      end
      kv.pull(key_ary, out: vals)

      vals.each do |vv|
        vv.each do |v|
          check_diff_to_scalar(v, num_devs * num_push)
        end
      end
    end

    specify do
      kv = init_kv
      kv.updater = updater
      check_updater(kv, 3, keys)
    end

    specify do
      kv = init_kv_with_str
      kv.updater = str_updater
      check_updater(kv, 'a', str_keys)
    end
  end

  describe '#init' do
    def check_init(kv, key)
      kv.init(key, MXNet::NDArray.ones(shape) * 4)
      a = MXNet::NDArray.zeros(shape)
      kv.pull(key, out: a)
      check_diff_to_scalar(a, 4)
    end

    specify do
      check_init(MXNet::KVStore.new, 3)
    end

    specify do
      check_init(MXNet::KVStore.new, 'a')
    end

    # TODO: support init by hash
  end

  describe '#type' do
    specify do
      kv_type = 'local_allreduce_cpu'
      expect(MXNet::KVStore.new(kv_type).type).to eq(kv_type)
    end
  end

  xdescribe 'invalid pull' # TODO:
end
