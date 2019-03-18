## tf_logits.py -- end-to-end differentable text-to-speech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import sys
sys.path.append("DeepSpeech")

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav

# Okay, so this is ugly. We don't want DeepSpeech to crash
# when we haven't built the language model.
# So we're just going to monkeypatch TF and make it a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
import DeepSpeech
os.path.exists = tmp

def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)

    # 1. Pre-emphasizer, a high-pass filter
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,1000),dtype=np.float32)), 1)

    # 2. windowing into frames of 320 samples, overlapping
    windowed = tf.stack([audio[:, i:i+400] for i in range(0,size-320,160)],1)

    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2)+1e-30
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+1e-30

    # 5. Take the DCT again, because why not
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]

    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]

    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
    
    return feat

                                          
def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)
        # Okay, so this is ugly again.
        # We just want it to not crash.
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        # Stop the annoying warnings
        tf.app.flags.FLAGS.decoder_library_path = "."
        DeepSpeech.initialize_globals()

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # (this is differentable with our implementation above)
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+19*26] for i in range(0,features.shape[1]-19*26+1,26)],1)
    features = tf.reshape(features, [batch_size, -1, 19*26])

    # 3. Whiten the data
    mean, var = tf.nn.moments(features, axes=[0,1,2])
    features = (features-mean)/(var**.5)

    # 4. Finally we process it with DeepSpeech
    logits = DeepSpeech.BiRNN(features, length, [0]*10)

    return logits

