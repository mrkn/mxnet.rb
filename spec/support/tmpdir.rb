require "tmpdir"

RSpec.shared_context 'within tmpdir', :within_tmpdir do
  around do |this_example|
    Dir.mktmpdir do |path|
      Dir.chdir(path) do
        this_example.run
      end
    end
  end
end
