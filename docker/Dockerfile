#FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    wget \
    vim \
    python3-pip \
    cmake cmake-curses-gui \
    libboost-dev libcgal-dev libcgal-qt5-dev \
    libblas-dev liblapack-dev libglu1-mesa-dev xorg-dev \
 	libmysqlclient-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Create a non-root user and switch to it
ARG UID
RUN adduser --disabled-password  -u $UID --gecos '' --shell /bin/bash docker
RUN echo "docker ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /work
USER docker


ENTRYPOINT ["/bin/bash"]
