Welcome to the ruby mxnet bindings with access to core mxnet modules including NDArray and Gluon

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'mxnet'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install mxnet

## Usage
Make sure that the mxnet library files are available by including the .so in $LD_LIBRARY_PATH environment variable.
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to mxnet library>`

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/mrkn/mxnet.

### NOTE: Don't use RuboCop in this project

We don't want to use RuboCop because we can manage our coding style by ourselves.  We want to accept small fluctuations in our coding style because we use Ruby.
Please do not submit issues and PRs that aim to introduce RuboCop in this repository.

### NOTE: Don't contaminate any styling changes in your pull-requests

We want any pull-requests to consist of changes only that are essential for the purposes of the pull-requests.
Please do not contaimnate any styling changes in your pull-requests.

## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
