require 'spec_helper'

RSpec.describe MXNet::Name::NameManager do
  subject(:name_manager) do
    MXNet::Name::NameManager.new
  end

  describe '#get' do
    specify do
      expect(subject.get('foo', 'bar')).to eq('foo')
      expect(subject.get(nil, 'bar')).to eq('bar0')
      expect(subject.get(nil, 'bar')).to eq('bar1')
      expect(subject.get(nil, 'baz')).to eq('baz0')
    end
  end

  describe '#enter and #exit' do
    specify do
      old_name_manager = MXNet::Name::NameManager.current
      subject.enter
      expect(MXNet::Name::NameManager.current).to equal(subject)
      subject.exit
      expect(MXNet::Name::NameManager.current).to equal(old_name_manager)
    end
  end

  describe '#enter with block' do
    specify do
      old_name_manager = MXNet::Name::NameManager.current
      expect { |b|
        subject.enter do
          b.to_proc.call
          expect(MXNet::Name::NameManager.current).to equal(subject)
        end
      }.to yield_control
      expect(MXNet::Name::NameManager.current).to equal(old_name_manager)
    end
  end
end

RSpec.describe MXNet::Name::Prefix do
  subject(:name_manager) do
    MXNet::Name::Prefix.new(prefix: prefix)
  end

  let(:prefix) { 'test_' }

  specify do
    expect(MXNet::Name::Prefix).to be < MXNet::Name::NameManager
  end

  describe '#get' do
    specify do
      expect(subject.get('foo', 'bar')).to eq('test_foo')
      expect(subject.get(nil, 'bar')).to eq('test_bar0')
      expect(subject.get(nil, 'bar')).to eq('test_bar1')
      expect(subject.get(nil, 'baz')).to eq('test_baz0')
    end
  end
end
