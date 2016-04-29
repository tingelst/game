FROM ubuntu:16.04

MAINTAINER Lars Tingelstad

RUN apt-get update && apt-get -y install \
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
    && rm -rf /var/lib/apt/lists/*

RUN pip install jupyter

RUN mkdir -p /usr/src/ \
    && curl -SL http://ceres-solver.org/ceres-solver-1.11.0.tar.gz \
    | tar -xvzC /usr/src/ \
    && mkdir -p /usr/src/ceres-solver-1.11.0/build \
    && cd /usr/src/ceres-solver-1.11.0/build \
    && cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF .. \
    && make \
    && make install

VOLUME /home/game/

EXPOSE 8888



WORKDIR /home/game/


# RUN pip install jupyter
# RUN mkdir -p -m 700 /root/.jupyter/ && \
#     echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py

# WORKDIR /usr/src/game/python
# ENTRYPOINT ["jupyter", "notebook"]




