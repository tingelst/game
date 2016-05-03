FROM ubuntu:16.04

MAINTAINER Lars Tingelstad

RUN apt-get update && apt-get -y install \
    wget \
    curl \
    git \
    cmake \
    build-essential \
    python-pip \
    libpython-dev \
    python-numpy \
    python-matplotlib \
    python-scipy \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    clang-3.8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install jupyter

# Install Ceres
RUN mkdir -p /usr/src/ \
    && curl -SL http://ceres-solver.org/ceres-solver-1.11.0.tar.gz \
    | tar -xvzC /usr/src/ \
    && mkdir -p /usr/src/ceres-solver-1.11.0/build \
    && cd /usr/src/ceres-solver-1.11.0/build \
    && cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF .. \
    && make -j12 \
    && make install

RUN cd /usr/src/ \
    && git clone https://github.com/google/benchmark.git \
    && mkdir -p /usr/src/benchmark/build/ \
    && cd /usr/src/benchmark/build/ \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_LTO=true .. \
    && make -j12 \
    && make install

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.9.0/tini \
    && echo "faafbfb5b079303691a939a747d7f60591f2143164093727e870b289a44d9872 *tini" | sha256sum -c - \
    && mv tini /usr/local/bin/tini \
    && chmod +x /usr/local/bin/tini

RUN useradd -m -s /bin/bash -N -u 1000 game 

USER game

RUN mkdir /home/game/game/ \
    && mkdir /home/game/.jupyter

USER root

VOLUME /home/game/game

EXPOSE 8888
WORKDIR /home/game/game/python
ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "notebook"]

# Add local files as late as possible to avoid cache busting
# Start notebook server
COPY jupyter_notebook_config.py /home/game/.jupyter/
RUN chown -R game:users /home/game/.jupyter

# Switch back to user to avoid accidental container runs as root
USER game
