require 'digest/sha1'
require 'open-uri'

module MXNet
  module Utils
    module_function

    def dtype_id(name_or_id)
      case name_or_id
      when String, ::Symbol
        MXNet::DType.name2id(name_or_id)
      when Integer
        name_or_id
      else
        raise TypeError, "wrong type of dtype: #{name_or_id.class} (expected Symbol or Integer)"
      end
    end

    def dtype_name(name_or_id)
      case name_or_id
      when String, ::Symbol
        name_or_id.to_s
      when Integer
        MXNet::DType.id2name(name_or_id)
      else
        raise TypeError, "wrong type of dtype: #{name_or_id.class} (expected Symbol or Integer)"
      end
    end

    # Check whether the sha1 hash of the file content matches the expected hash.
    def check_sha1(filename, sha1_hash)
      actual_hash = Digest::SHA1.hexdigest(::IO.binread(filename))
      return actual_hash == sha1_hash
    end

    # Download an given URL
    def download(url, path: nil, overwrite: false, sha1_hash: nil)
      if path.nil?
        fname = url.split('/')[-1]
      else
        path = File.expand_path(path)
        if File.directory?(path)
          fname = File.join(path, url.split('/')[-1])
        else
          fname = path
        end
      end

      if overwrite || !File.exist?(fname) || (sha1_hash && !check_sha1(fname, sha1_hash))
        dirname = File.dirname(fname)
        FileUtils.mkdir_p(dirname)

        puts "Downloading #{fname} from #{url}..."
        open(url, 'rb') {|src| ::IO.copy_stream(src, fname) }
        if sha1_hash && !check_sha1(fname, sha1_hash)
          raise "File #{fname} is downloaded but the content hash does not match. " +
                "The repo may be outdated or download may be incomplete. " +
                "If the `repo_url` is overridden, consider switching to " +
                "the default repo."
        end
      end

      return fname
    end

    DEFAULT_REPO = -"https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/"

    # REturn the base URL for Gluon dataset and model repository.
    def get_repo_url
      repo_url = ENV['MXNET_GLUON_REPO'] || DEFAULT_REPO
      repo_url += '/' if repo_url[-1] != '/'
      repo_url
    end

    # Return the URL for hosted file in Gluon repository.
    def get_repo_file_url(namespace, filename)
      "#{get_repo_url}#{namespace}/#{filename}"
    end
  end
end
