#!/bin/bash

rm_test_dir() {
    if [[ $(pwd) =~ '/audio_adversarial_examples$' ]]
    then
        rm -rf audio_adversarial_examples
    fi
}

rm_test_dir
docker image rm aae_deepspeech_041_gpu aae_deepspeech_041_cpu
git clone git@github.com:tom-doerr/audio_adversarial_examples.git
cd audio_adversarial_examples
./setup.sh
docker run --gpus all -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_gpu bash -c '
python3 classify.py --in sample-000000.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1 &&\
python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1 &&\
python3 classify.py --in adv.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
' &&\
rm_test_dir

