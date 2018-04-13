require 'spec_helper'

RSpec.describe MXNet::Registry::Manager do
  let(:namespace) { Module.new }

  before do
    class namespace::TestClass
      def initialize(*args, **kwargs)
        @args = args
        @params = kwargs
      end

      attr_reader :args, :params
    end

    class namespace::TestSubclass1 < namespace::TestClass
    end

    class namespace::TestSubClass1 < namespace::TestClass
    end

    class namespace::TestSubclass2 < namespace::TestClass
    end

    class namespace::TestOtherClass
    end
  end

  subject(:regman) do
    MXNet::Registry::Manager.new(namespace::TestClass, :test_class)
  end

  let(:internal_registry) do
    regman.instance_variable_get(:@registry)
  end

  describe '#register' do
    specify do
      expect(regman.register(namespace::TestSubclass1)).to equal(namespace::TestSubclass1)
      expect(regman.register(namespace::TestSubclass2)).to equal(namespace::TestSubclass2)
    end

    specify do
      regman.register(namespace::TestSubclass1)
      expect(internal_registry.length).to eq(1)
      expect(internal_registry.keys).to eq([:testsubclass1])

      regman.register(namespace::TestSubclass1)
      expect(internal_registry.length).to eq(1)
      expect(internal_registry.keys).to eq([:testsubclass1])

      regman.register(namespace::TestSubClass1)
      expect(internal_registry.length).to eq(1)
      expect(internal_registry.keys).to eq([:testsubclass1])
      expect(internal_registry[:testsubclass1]).to equal(namespace::TestSubClass1)

      regman.register(namespace::TestSubClass1, :foo)
      expect(internal_registry.length).to eq(2)
      expect(internal_registry.keys).to eq([:testsubclass1, :foo])
      expect(internal_registry[:testsubclass1]).to equal(namespace::TestSubClass1)
      expect(internal_registry[:foo]).to equal(namespace::TestSubClass1)

      regman.register(namespace::TestSubclass2, :foo)
      expect(internal_registry.length).to eq(2)
      expect(internal_registry.keys).to eq([:testsubclass1, :foo])
      expect(internal_registry[:testsubclass1]).to equal(namespace::TestSubClass1)
      expect(internal_registry[:foo]).to equal(namespace::TestSubclass2)

      expect { regman.register(namespace::TestOtherClass) }.to raise_error(TypeError)
      expect(internal_registry.length).to eq(2)
      expect(internal_registry.keys).to eq([:testsubclass1, :foo])
      expect(internal_registry[:testsubclass1]).to equal(namespace::TestSubClass1)
      expect(internal_registry[:foo]).to equal(namespace::TestSubclass2)
    end
  end

  describe '#alias' do
    specify do
      expect(regman.alias(namespace::TestSubclass1, :bar, :baz)).to equal(namespace::TestSubclass1)
      expect(internal_registry.length).to eq(2)
      expect(internal_registry.keys).to eq([:bar, :baz])
    end

    specify do
      expect(regman.alias(namespace::TestSubclass1, :CamelCase)).to equal(namespace::TestSubclass1)
      expect(internal_registry.length).to eq(1)
      expect(internal_registry.keys).to eq([:camelcase])
    end
  end

  describe '#create' do
    before do
      regman.register(namespace::TestSubclass1)
      regman.register(namespace::TestSubclass2, :foo)
    end

    specify 'name only' do
      expect(regman.create(:testsubclass1)).to be_a(namespace::TestSubclass1)
      expect(regman.create(:testsubclass1, a: 1, b: 2).params).to eq({a: 1, b: 2})
      expect(regman.create(:TESTSUBCLASS1)).to be_a(namespace::TestSubclass1)

      expect(regman.create(:foo)).to be_a(namespace::TestSubclass2)
      expect(regman.create(:foo, 1, 2).args).to eq([1, 2])
      expect(regman.create(:foo, c: 2, d: 3).params).to eq({c: 2, d: 3})
      expect(regman.create(:FOO)).to be_a(namespace::TestSubclass2)

      expect(regman.create(test_class: :TestSubclass1)).to be_a(namespace::TestSubclass1)
      expect(regman.create(test_class: :foo)).to be_a(namespace::TestSubclass2)
      expect(regman.create(test_class: :FOO)).to be_a(namespace::TestSubclass2)

      expect { regman.create(:unknown_name) }.to raise_error(ArgumentError)
      expect { regman.create(1) }.to raise_error(ArgumentError)
    end

    specify 'instance' do
      x = namespace::TestSubclass1.new
      expect(regman.create(x)).to equal(x)
      expect { regman.create(x, 1) }.to raise_error(ArgumentError)
      expect { regman.create(x, a: 1) }.to raise_error(ArgumentError)
    end


    specify 'hash' do
      expect(regman.create({test_class: :foo, a: 1})).to be_a(namespace::TestSubclass2)
      expect(regman.create({test_class: :FOO, a: 1})).to be_a(namespace::TestSubclass2)
    end

    specify 'json' do
      expect(regman.create([:foo].to_json)).to be_a(namespace::TestSubclass2)
      expect(regman.create([:foo, {a: 1, b: 2}].to_json).params).to eq({a: 1, b: 2})
      expect(regman.create({test_class: :foo}.to_json)).to be_a(namespace::TestSubclass2)
      expect(regman.create({test_class: :foo, a: 1, b: 2}.to_json).params).to eq({a: 1, b: 2})
    end
  end
end
