# coding: utf-8
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "mxnet/version"

Gem::Specification.new do |spec|
  spec.name          = "mxnet"
  spec.version       = MXNet::VERSION
  spec.authors       = ["Kenta Murata"]
  spec.email         = ["mrkn@mrkn.jp"]

  spec.summary       = %q{MXNet binding for Ruby}
  spec.description   = %q{MXNet binding for Rub}
  spec.homepage      = "https://github.com/mrkn/mxnet.rb"
  spec.license       = "MTL"

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/mxnet/extconf.rb", "ext/mxnet/narray_helper/extconf.rb"]

  spec.add_dependency "fiddle"
  spec.add_dependency "numo-narray"

  spec.add_development_dependency "bundler", ">= 2.1.2"
  spec.add_development_dependency "rake", ">= 12.0"
  spec.add_development_dependency "rake-compiler"
  spec.add_development_dependency "rspec", ">= 3.8"
  spec.add_development_dependency "pry"
end
