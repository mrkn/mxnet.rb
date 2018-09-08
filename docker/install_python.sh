#!/bin/bash

set -ex

wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"
wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"

export GNUPGHOME="$(mktemp -d)"
gpg --keyserver pool.sks-keyservers.net --recv-keys "$PYTHON_GPG_KEY"
gpg --batch --verify python.tar.xz.asc python.tar.xz
{ command -v gpgconf > /dev/null && gpgconf --kill all || :; }
rm -rf "$GNUPGHOME" python.tar.xz.asc
mkdir -p /usr/src/python
tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz
rm python.tar.xz

cd /usr/src/python
gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"
./configure \
     --build="$gnuArch" \
     --enable-loadable-sqlite-extensions \
     --enable-shared \
     --with-system-expat \
     --with-system-ffi \
     --without-ensurepip
make -j "$(nproc)"
make install
ldconfig

find /usr/local -depth \
     \( \
     	\( -type d -a \( -name test -o -name tests \) \) \
     	-o \
     	\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
     \) -exec rm -rf '{}' +
rm -rf /usr/src/python

python3 --version
