require "bundler"
Bundler::GemHelper.install_tasks

require "rake"
require "rake/extensiontask"
require "rspec/core/rake_task"

Dir[File.expand_path('../tasks/**/*.rake', __FILE__)].each {|f| load f }

Rake::ExtensionTask.new('mxnet')
Rake::ExtensionTask.new('mxnet/narray_helper')
RSpec::Core::RakeTask.new(:spec)

task :default => :spec
task spec: :compile
