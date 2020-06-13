This is the code corresponding to the paper
"Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"
Nicholas Carlini and David Wagner
https://arxiv.org/abs/1801.01944

To generate adversarial examples for your own files, follow the below process
and modify the arguments to attack,py. Ensure that the file is sampled at
16KHz and uses signed 16-bit ints as the data type. You may want to modify
the number of iterations that the attack algorithm is allowed to run.

WARNING: THIS IS NOT THE CODE USED IN THE PAPER. If you just want to get going
generating adversarial examples on audio then proceed as described below.

The current master branch points to code which will run on TensorFlow 1.14 and
DeepSpeech 0.4.1, an almost-recent version of the dependencies. (Large portions
of tf_logits.py will need to be re-written to run on DeepSpeech 0.5.1 which uses
a new feature extraction pipeline with TensorFlow's C++ implementation. If you
feel motivated to do that I would gladly accept a PR.)

However, IF YOU ARE TRYING TO REPRODUCE THE PAPER (or just have decided
that you enjoy pain and want to suffer through dependency hell) then you
will have to checkout commit a8d5f675ac8659072732d3de2152411f07c7aa3a and
follow the README from there.



# Installation
1. Install Docker
https://docs.docker.com/install/

2. Download DeepSpeech and build the Docker images:
```
$ ./setup.sh
```

### With Nvidia-GPU support:
3. Start the container using the GPU image we just build
```
$ docker run --gpus all -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_gpu
```

### CPU-only (Skip if already started with Nvidia-GPU support):
3. Start the container using the CPU image we just build
```
$ docker run -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_cpu
```


### Test Setup
4. Check that you can classify normal audio correctly
```
$ python3 classify.py --in sample-000000.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

5. Generate adversarial examples
```
$ python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

6. Verify the attack succeeded
```
$ python3 classify.py --in adv.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

# Docker Hub
The docker images are available on Docker Hub.

CPU-Version: `tomdoerr/aae_deepspeech_041_cpu`

GPU-Version: `tomdoerr/aae_deepspeech_041_gpu`


