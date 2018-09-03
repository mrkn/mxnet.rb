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

namespace :ci do
  def get_image_name
    ruby_version = ENV['ruby_version'] || '2.5.1'
    python_version = ENV['python_version'] || '3.7.0'
    mxnet_version = ENV['mxnet_version'] || '1.2.1.post1'
    return ['mrkn/mxnet-rb-ci', [ruby_version, python_version, mxnet_version].join('-')].join(':')
  end

  task :pull do
    sh 'docker', 'pull', get_image_name
  end

  task :run do
    sh 'docker', 'run', '--rm', '-t', '-v', "#{Dir.pwd}:/tmp", get_image_name, '/bin/sh', '-c', 'cd /tmp; bin/ci-script'
  end
end
