#!/bin/bash

if [[ "$1" == "--gpu" ]]
then
    docker build -t "aae_deepspeech_041_gpu"  - < docker/aae_deepspeech_041_gpu.dockerfile
elif [[ "$1" == "--cpu" ]]
then
    docker build -t "aae_deepspeech_041_cpu"  - < docker/aae_deepspeech_041_cpu.dockerfile
fi

git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
git checkout v0.4.1
cd ..
