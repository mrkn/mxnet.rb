#!/bin/bash

set -ex

RUBY_VERSION=${RUBY_VERSION-2.6.6}
RUBY_MAJOR=$(echo -n $RUBY_VERSION | sed -E 's/\.[0-9]+(-.*)?$//g')
RUBYGEMS_VERSION=${RUBYGEMS_VERSION-2.7.7}
BUNDLER_VERSION=${BUNDLER_VERSION-2.1.2}

case $RUBY_VERSION in
  2.6.0)
    RUBY_DOWNLOAD_SHA256=acb00f04374899ba8ee74bbbcb9b35c5c6b1fd229f1876554ee76f0f1710ff5f
    ;;
  2.5.3)
    RUBY_DOWNLOAD_SHA256=1cc9d0359a8ea35fc6111ec830d12e60168f3b9b305a3c2578357d360fcf306f
    ;;
  2.4.5)
    RUBY_DOWNLOAD_SHA256=2f0cdcce9989f63ef7c2939bdb17b1ef244c4f384d85b8531d60e73d8cc31eeb
    ;;
  2.3.8)
    RUBY_DOWNLOAD_SHA256=910f635d84fd0d81ac9bdee0731279e6026cb4cd1315bbbb5dfb22e09c5c1dfe
    ;;
  *)
    echo "Unsupported RUBY_VERSION ($RUBY_VERSION)" >2
    exit 1
    ;;
esac

buildDeps=$(cat /tmp/ruby_build_dep.txt)

apt-get update
apt-get install -y --no-install-recommends $buildDeps
# Need to down grade openssl to 1.0.x for Ruby 2.3.x
case $RUBY_VERSION in
  2.3.*)
    apt-get install -y --no-install-recommends libssl1.0-dev
    ;;
esac
rm -rf /var/lib/apt/lists/*

wget -O ruby.tar.xz "https://cache.ruby-lang.org/pub/ruby/${RUBY_MAJOR}/ruby-${RUBY_VERSION}.tar.xz"
echo "$RUBY_DOWNLOAD_SHA256 *ruby.tar.xz" | sha256sum -c -

mkdir -p /usr/src/ruby
tar -xJf ruby.tar.xz -C /usr/src/ruby --strip-components=1
rm ruby.tar.xz

cd /usr/src/ruby

{
  echo '#define ENABLE_PATH_CHECK 0'
  echo
  cat file.c
} > file.c.new
mv file.c.new file.c

autoconf
gnuArch=$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)
./configure \
   --build="$gnuArch" \
   --disable-install-doc \
   --enable-shared

make -j "$(nproc)"
make install

dpkg-query --show --showformat '${package}\n' \
  | grep -P '^libreadline\d+$' \
  | xargs apt-mark manual
cd /
rm -r /usr/src/ruby

gem update --system "$RUBYGEMS_VERSION"
gem install bundler --version "$BUNDLER_VERSION" --force
rm -r /root/.gem/
