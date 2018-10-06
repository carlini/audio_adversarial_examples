# [Audio Adversarial Examples: Targeted Attacks on Speech-to-Text](https://arxiv.org/pdf/1801.01944.pdf)
### Authors: Nicholas Carlini and David Wagner

This release of the code is preliminary; it includes the CTC-based attack with a batch size of 1.
This means it requires a slightly larger distortion, and is slower to run than the algorithm that is presented in the paper.
Soon™ I will add these two improvements to this codebase.

To generate adversarial examples for your own files, follow the below process and modify the arguments to **attack.py**

Ensure that the file is sampled at **16KHz**.
You may want to modify the number of iterations that the attack algorithm is allowed to run.


### Instructions for basic use:

1. Install the dependencies

    ```pip3 install --user numpy scipy tensorflow-gpu pandas python_speech_features```

2. Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech:

    ```git clone https://github.com/mozilla/DeepSpeech.git```

3. Checkout the correct version of the code:

    ``` cd DeepSpeech; git checkout tags/v0.1.1```

4. Download the DeepSpeech model

    ```wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz```
    ```tar -xzf deepspeech-0.1.0-models.tar.gz```

5. Verify that you have a file models/output_graph.pb, it's MD5 sum should be
    ```08a9e6e8dc450007a0df0a37956bc795.```

6. Convert the .pb to a TensorFlow checkpoint file

    ```python3 make_checkpoint.py```

7. Generate adversarial examples

    ```python3 attack.py --in sample.wav --target "example" --out adversarial.wav```

8. (optional) Install the deepseech utility:

    ```pip3 install deepspeech-gpu```

9. Classify the generated phrase

    ```deepspeech models/output_graph.pb adversarial.wav models/alphabet.txt```
