FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# >> START Install base software

# Get basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++ \
        gcc \
        git \
        libbz2-dev \
        libboost-all-dev \
        libgsm1-dev \
        libltdl-dev \
        liblzma-dev \
        libmagic-dev \
        libpng-dev \
        libsox-fmt-mp3 \
        libsox-dev \
        locales \
        openjdk-8-jdk \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-numpy \
        sox \
        unzip \
        wget \
        zlib1g-dev

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

ENV PYTHON_BIN_PATH /usr/bin/python3.6
ENV PYTHON_LIB_PATH /usr/local/lib/python3.6/dist-packages

WORKDIR /

# Get DeepSpeech with the version 0.9.3
RUN git clone --branch v0.9.3 https://github.com/mozilla/DeepSpeech
RUN curl -sSL https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-checkpoint.tar.gz \
| tar xvz 
# Get the scorer for CTC decoder
RUN curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# Copy required files 
COPY attack.py .
COPY classify.py .
COPY tf_logits.py .
COPY mfcc.py .
COPY sample-000000.wav .

WORKDIR /DeepSpeech

# Install DeepSpeech dependencies
RUN pip3 install --upgrade pip==20.2.2 wheel==0.34.2 setuptools==49.6.0
RUN pip3 install --upgrade -e .
# Remove the numpy that causes dependency issues
RUN pip3 uninstall -y numpy
# Install the right version of numpy and the remaining packages
RUN pip3 install numpy==1.18.5 \
    python_speech_features \
    tables 
RUN apt-get install python3-dev
# Install TensorFlow with GPU support 
RUN pip3 uninstall -y tensorflow
RUN pip3 install 'tensorflow-gpu==1.15.4'

# Install vim for light editing
RUN apt-get -y install vim

# Done
WORKDIR /