FROM buildpack-deps:bionic

ARG RUBY_VERSION=2.5.1
ARG PYTHON_VERSION=3.7.0
ARG MXNET_VERSION=1.2.1.post1

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Ruby build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            bzip2 \
            ca-certificates \
            libffi-dev \
            libgmp-dev \
            libgdbm-dev \
            libssl-dev \
            libyaml-dev \
            procps \
            zlib1g-dev \
            libgmp-dev \
            && \
    rm -rf /var/lib/apt/lists/*

# skip installing gem documentation
RUN mkdir -p /usr/local/etc && \
    { \
      echo 'install: --no-document'; \
      echo 'update: --no-document'; \
    } >> /usr/local/etc/gemrc

ENV RUBY_VERSION $RUBY_VERSION
ENV RUBYGEMS_VERSION 2.7.7
ENV BUNDLER_VERSION 1.16.4

ADD ruby_build_dep.txt /tmp
ADD install_ruby.sh /tmp
RUN /tmp/install_ruby.sh

ENV GEM_HOME /usr/local/bundle
ENV BUNDLE_PATH="$GEM_HOME" \
    BUNDLE_SILENCE_ROOT_WARNING=1 \
    BUNDLE_APP_CONFIG="$GEM_HOME"
ENV PATH $GEM_HOME/bin:$BUNDLE_PATH/gems/bin:$PATH
RUN mkdir -p "$GEM_HOME" && chmod 777 "$GEM_HOME"

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

RUN apt-get purge -y --auto-remove $(cat /tmp/ruby_build_dep.txt)
RUN rm /tmp/*
