# Audio Adversarial Examples
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

There are two ways to install this project. The first is to just use Docker
with a buildfile provided by Tom Doerr. It works. The second is to try and
set up everything on your machine directly. This might work, if you happen
to have the right versions of things.


# Docker Installation (highly recommended)

These docker instructions were kindly provided by Tom Doerr, and are simple to follow if you have Docker set up.


1. Install Docker.
On Ubuntu/Debian/Linux-Mint etc.:
```
sudo apt-get install docker.io
sudo systemctl enable --now docker
```
Instructions for other platforms:
https://docs.docker.com/install/


2. Download DeepSpeech and build the Docker images:
```
$ ./setup.sh
```

### With Nvidia-GPU support:
3. Install the NVIDIA Container Toolkit.
This step will only work on Linux and is only necessary if you want GPU support.
As far as I know it's not possible to use a GPU with docker under Windows/Mac.
On Ubuntu/Debian/Linux-Mint etc. you can install the toolkit with the following commands:
```sh
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
Instructions for other platforms (CentOS/RHEL):
https://github.com/NVIDIA/nvidia-docker

4. Start the container using the GPU image we just build:
```
$ docker run --gpus all -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_gpu
```

### CPU-only (Skip if already started with Nvidia-GPU support):
4. Start the container using the CPU image we just build:
```
$ docker run -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_cpu
```


### Test Setup
5. Check that you can classify normal audio correctly:
```
$ python3 classify.py --in sample-000000.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

6. Generate adversarial examples:
```
$ python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

7. Verify the attack succeeded:
```
$ python3 classify.py --in adv.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

## Docker Hub
The docker images are available on Docker Hub.

CPU-Version: `tomdoerr/aae_deepspeech_041_cpu`

GPU-Version: `tomdoerr/aae_deepspeech_041_gpu`



# Direct Install

These are the original instructions from earlier. They will work, but require manual installs.


1. Install the dependencies
```
pip3 install tensorflow-gpu==1.14 progressbar numpy scipy pandas python_speech_features tables attrdict pyxdg
pip3 install $(python3 util/taskcluster.py --decoder)
```

Download and install
https://git-lfs.github.com/

1b. Make sure you have installed git lfs. Otherwise later steps will mysteriously fail.

2. Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech:
```
git clone https://github.com/mozilla/DeepSpeech.git
```

2b. Checkout the correct version of the code:
```
(cd DeepSpeech; git checkout tags/v0.4.1)
```

2c. If you get an error with tflite_convert, comment out DeepSpeech.py Line 21
```
from tensorflow.contrib.lite.python import tflite_convert
```

3. Download the DeepSpeech model

```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
tar -xzf deepspeech-0.4.1-checkpoint.tar.gz
```

4. Verify that you have a file deepspeech-0.4.1-checkpoint/model.v0.4.1.data-00000-of-00001
Its MD5 sum should be
```
ca825ad95066b10f5e080db8cb24b165
```

5. Check that you can classify normal images correctly
```
python3 attack.py --in sample-000000.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

6. Generate adversarial examples
```
python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

8. Verify the attack succeeded
```
python3 attack.py --in adv.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```
