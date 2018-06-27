module MXNet
  module LibMXNet
    module Finder
      case RUBY_PLATFORM
      when /cygwin/
        libprefix = 'cyg'
        libsuffix = 'dll'
      when /mingw/, /mswin/
        libprefix = ''
        libsuffix = 'dll'
      when /darwin/
        libsuffix = 'dylib'
      end

      LIBPREFIX = libprefix || 'lib'
      LIBSUFFIX = libsuffix || 'so'

      module_function

      def find_libmxnet
        top_dir = File.expand_path('../../../..', __FILE__)
        lib_dir = File.join(top_dir, 'lib')
        dll_path = [lib_dir]
        if RUBY_PLATFORM =~ /(?:mingw|mswin|cygwin)/i
          ENV['PATH'].split(';').each do |path|
            dll_path << path.strip
          end
          ENV['PATH'] = "#{lib_dir};#{ENV['PATH']}"
        end
        if RUBY_PLATFORM !~ /(?:mswin|darwin)/i
          if ENV['LD_LIBRARY_PATH']
            ENV['LD_LIBRARY_PATH'].split(':').each do |path|
              dll_path << path.strip
            end
          end
        end
        dll_path.map! {|path| File.join(path, "#{LIBPREFIX}mxnet.#{LIBSUFFIX}") }
        dll_path.unshift(find_libmxnet_in_python(ENV['PYTHON']))
        dll_path.unshift(ENV['LIBMXNET'])
        dll_path.compact!
        lib_path = dll_path.select { |path| File.file?(path) }
        if lib_path.empty?
          raise "Unable to find MXNet shared library.  The list of candidates:\n#{dll_path.join("\n")}"
        end
        lib_path
      end

      DEFAULT_PYTHON_COMMANDS = [
        -'python3',
        -'python',
      ].freeze

      def find_libmxnet_in_python(python=nil)
        python ||= DEFAULT_PYTHON_COMMANDS
        investigator_py = File.expand_path('../find_libmxnet.py', __FILE__)
        Array(python).each do |python_command|
          lib_path = `'#{python_command}' '#{investigator_py}' 2>/dev/null`.chomp
          return lib_path if $?.success?
        end
        nil
      end
    end
  end
end
