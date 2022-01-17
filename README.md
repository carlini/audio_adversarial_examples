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

The current master branch points to code which runs on TensorFlow 1.15.4 and 
DeepSpeech 0.9.3, which is the most current version of DeepSpeech. If you want 
your code to run on TensorFlow 1.14 and DeepSpeech 0.4.1, then checkout 
commit e0e0486bf1f370582a0845236285cb2a8f8a9f7f. 

However, IF YOU ARE TRYING TO REPRODUCE THE PAPER (or just have decided
that you enjoy pain and want to suffer through dependency hell) then you
will have to checkout commit a8d5f675ac8659072732d3de2152411f07c7aa3a and
follow the README from there.

# Running the Attack on DeepSpeech 
Currently, we only have the docker image that runs on gpu. Please make sure that you have gpu support. DeepSpeech v0.9.3 has dependency on CUDA 10.1 and CuDNN v7.6.

1. Install Docker. (skip this step if you already have docker)
On Ubuntu/Debian/Linux-Mint etc.:
```
sudo apt-get install docker.io
sudo systemctl enable --now docker
```
Instructions for other platforms:
https://docs.docker.com/install/

2. Build the docker image (Make sure that you have gpu support, if not the image won't run)
```
docker build -t aae_deepspeech_093_gpu .
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

4. Run the image. Please specify the absolute path to your data and tmp folder (create one if you don't have). The data folder should contain all your audio data. The tmp folder is used to store your results.
```
docker run --gpus all \
-v /absolute/path/to/data:/data \
-v /absolute/path/to/tmp:/tmp \
-ti aae_deepspeech_093_gpu
```
If you have no data and just want to test if it works, run
```
docker run --gpus all -ti aae_deepspeech_093_gpu
```

5. Check that you can classify normal audio correctly:
```
python3 classify.py \
        --input sample-000000.wav \
        --restore_path deepspeech-0.9.3-checkpoint/best_dev-1466475 \
        --scorer_path deepspeech-0.9.3-models.scorer \
        --alphabet_config_path DeepSpeech/data/alphabet.txt \
```

6. Generate an adversarial example:
```
python3 attack.py \
        --input sample-000000.wav \
        --outprefix adv \
        --target "this is a test" \
        --iterations 1000 \
        --restore_path deepspeech-0.9.3-checkpoint/best_dev-1466475 \
        --scorer_path deepspeech-0.9.3-models.scorer \
        --alphabet_config_path DeepSpeech/data/alphabet.txt \
```

7. Verify that the attack succeeded:
```
python3 classify.py \
        --input adv0.wav \
        --restore_path deepspeech-0.9.3-checkpoint/best_dev-1466475 \
        --scorer_path deepspeech-0.9.3-models.scorer \
        --alphabet_config_path DeepSpeech/data/alphabet.txt \
```