#!/bin/bash

set -ex

RUBY_VERSION=${RUBY_VERSION-2.5.1}
RUBY_MAJOR=$(echo -n $RUBY_VERSION | sed -E 's/\.[0-9]+(-.*)?$//g')
RUBYGEMS_VERSION=${RUBYGEMS_VERSION-2.7.7}
BUNDLER_VERSION=${BUNDLER_VERSION-1.16.4}

case $RUBY_VERSION in
  "2.6.0-preview2")
    RUBY_DOWNLOAD_SHA256=00ddfb5e33dee24469dd0b203597f7ecee66522ebb496f620f5815372ea2d3ec
    ;;
  "2.5.1")
    RUBY_DOWNLOAD_SHA256=886ac5eed41e3b5fc699be837b0087a6a5a3d10f464087560d2d21b3e71b754d
    ;;
  "2.4.4")
    RUBY_DOWNLOAD_SHA256=1d0034071d675193ca769f64c91827e5f54cb3a7962316a41d5217c7bc6949f0
    ;;
  "2.3.7")
    RUBY_DOWNLOAD_SHA256=c61f8f2b9d3ffff5567e186421fa191f0d5e7c2b189b426bb84498825d548edb
    ;;
  *)
    echo "Unsupported RUBY_VERSION ($RUBY_VERSION)" >2
    exit 1
    ;;
esac

buildDeps=$(cat /tmp/ruby_build_dep.txt)

apt-get update
apt-get install -y --no-install-recommends $buildDeps
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
