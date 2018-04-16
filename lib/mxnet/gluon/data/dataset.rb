require 'mxnet/gluon/data'

module MXNet::Gluon::Data
  class Dataset
    include Enumerable

    def length
      raise NotImplementedError
    end

    def [](idx)
      raise NotImplementedError
    end

    def each
      return enum_for unless block_given?
      i = 0
      while i < length
        yield self[i]
        i += 1
      end
      self
    end

    def transform(lazy: true)
      trans = LazyTransformDataset.new(self, &proc)
      return trans if lazy
      SimpleDataset.new(p trans.map.to_a)
    end

    def transform_first(lazy: true)
      transform(lazy: lazy) {|x, *rest| [yield(x), *rest] }
    end
  end

  class SimpleDataset < Dataset
    def initialize(data)
      @data = data
    end

    def length
      @data.length
    end

    def [](idx)
      @data[idx]
    end
  end

  class LazyTransformDataset < Dataset
    def initialize(dataset)
      @dataset = dataset
      @fn = proc
    end

    def length
      @dataset.length
    end

    def [](idx)
      item = @dataset[idx]
      if item.is_a? Array
        @fn.(*item)
      else
        @fn.(item)
      end
    end

    def each
      return enum_for unless block_given?
      @dataset.each do |*args|
        yield @fn.(*args)
      end
      self
    end
  end

  class DownloadedDataset < Dataset
    def initialize(root:, transform:)
      super()
      @transform = transform
      @data = nil
      @label = nil
      root = File.expand_path(root)
      @root = -root
      FileUtils.mkdir_p(root) unless File.directory?(root)
      _get_data
    end

    attr_reader :root

    def length
      @label.length
    end

    def [](idx)
      if @transform
        @transform.(@data[idx], @label[idx])
      else
        [@data[idx], @label[idx]]
      end
    end

    private def _get_data
      raise NotImplementedError
    end
  end
end
