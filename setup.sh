#!/bin/bash

docker build -t "aae_deepspeech_041_gpu"  - < docker/aae_deepspeech_041_gpu.dockerfile
docker build -t "aae_deepspeech_041_cpu"  - < docker/aae_deepspeech_041_cpu.dockerfile
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
git checkout v0.4.1
cd ..
if [ ! -d deepspeech-0.4.1-checkpoint ]
then
    wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
    tar -xzf deepspeech-0.4.1-checkpoint.tar.gz
    rm deepspeech-0.4.1-checkpoint.tar.gz
fi
