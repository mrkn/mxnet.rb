require 'fiddle'

module MXNet
  module LibMXNet
    def self.load_lib
      require 'mxnet/libmxnet/finder'
      lib_path = Finder.find_libmxnet
      return Fiddle::Handle.new(lib_path[0], Fiddle::Handle::RTLD_LAZY)
    end

    def self.handle
      @handle ||= load_lib
    end
  end
end
