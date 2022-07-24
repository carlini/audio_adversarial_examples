import mfcc
from deepspeech_training.train import create_model, create_overlapping_windows
from deepspeech_training.util.config import Config
from deepspeech_training.util.flags import FLAGS
import DeepSpeech
import tensorflow as tf
import numpy as np
import sys

sys.path.append("DeepSpeech")

def periodic_hann_window(window_length, dtype):
    """
    Return periodic Hann window. Implementation based on:
    https://github.com/tensorflow/tensorflow/blob/bd962d8cdfcda01a23c7051fa05e3db86dd9c30f/tensorflow/core/kernels/spectrogram.cc#L28-L36
    """
    return 0.5 - 0.5 * tf.math.cos(2.0 * np.pi * tf.range(tf.to_float(window_length), dtype=dtype) / (tf.to_float(window_length)))


def get_logits(audio, length):
    """
    Compute the logits for a given waveform
    using functions from DeepSpeech v0.9.3.
    """
    # Scale audio to values between -1 and 1
    audio = tf.cast(audio / 2 ** 15, tf.float32)

    stfts = tf.signal.stft(
        audio,
        frame_length=512,
        frame_step=320,
        fft_length=512,
        window_fn=periodic_hann_window
    )
    spectrogram = tf.square(tf.abs(stfts))

    # Compute features
    features = mfcc.compute_mfcc(spectrogram=spectrogram, sample_rate=FLAGS.audio_sample_rate,
                                 upper_edge_hertz=FLAGS.audio_sample_rate / 2, dct_coefficient_count=Config.n_input)

    # Evaluate
    features = create_overlapping_windows(features)

    # Create DeepSpeech model
    no_dropout = [None] * 6
    logits, _ = create_model(features, seq_length=length,
                             dropout=no_dropout, overlap=False)

    return logits