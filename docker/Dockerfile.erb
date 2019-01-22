FROM rubylang/ruby:<%= ruby_version %>-bionic

ARG PYTHON_VERSION=3.7.2
ARG MXNET_VERSION=1.3.1

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Ruby build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            bzip2 \
            ca-certificates \
            curl \
            gcc \
            gnupg \
            libffi-dev \
            libgmp-dev \
            libssl-dev \
            libyaml-dev \
            make \
            procps \
            zlib1g-dev \
            unzip \
            wget \
            && \
    rm -rf /var/lib/apt/lists/*

# Python

# ensure local python is preferred over distributed python
ENV PATH /usr/local/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            tk-dev \
            uuid-dev \
            && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHON_GPG_KEY 0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D
ENV PYTHON_VERSION $PYTHON_VERSION

ADD install_python.sh /tmp
RUN /tmp/install_python.sh

# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin && \
    ln -s idle3 idle && \
    ln -s pydoc3 pydoc && \
    ln -s python3 python && \
    ln -s python3-config python-config

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 18.0
ADD install_pip.sh /tmp
RUN /tmp/install_pip.sh

# MXNet
ENV MXNET_VERSION=$MXNET_VERSION
RUN pip3 install "mxnet==$MXNET_VERSION"

RUN rm /tmp/*

RUN mkdir /work
WORKDIR /work

RUN mkdir -p /data/mnist && \
    cd /data/mnist && \
    curl -fsSL -O http://data.mxnet.io/mxnet/data/mnist.zip && \
    unzip -x mnist.zip && \
    rm /data/mnist/mnist.zip

RUN mkdir -p /data/cifar10 && \
    cd /data/cifar10 && \
    curl -fsSL -O http://data.mxnet.io/mxnet/data/cifar10.zip && \
    unzip -x cifar10.zip && \
    mv cifar/* . && \
    rm /data/cifar10/cifar10.zip
