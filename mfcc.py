import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal.shape_ops import frame

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def freq2mel(frequencies_hertz):
    """Convert frequency to mel frequency"""
    return _MEL_HIGH_FREQUENCY_Q * np.log1p(frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)


def _validate_arguments(filterbank_channel_count, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype):
    """Checks the inputs to linear_to_mel_weight_matrix."""
    if filterbank_channel_count <= 0:
        raise ValueError(
            'filterbank_channel_count must be positive. Got: %s' % filterbank_channel_count)

    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)

    if lower_edge_hertz < 0.0:
        raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                         lower_edge_hertz)

    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                         (lower_edge_hertz, upper_edge_hertz))

    if upper_edge_hertz > sample_rate / 2:
        raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                         'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                         % (upper_edge_hertz, sample_rate))

    if not dtype.is_floating:
        raise ValueError(
            'dtype must be a floating point type. Got: %s' % dtype)


def compute_mfcc_mel_filterbank(spectrogram, sample_rate=16000, lower_edge_hertz=20.0,
                                upper_edge_hertz=8000.0, filterbank_channel_count=40, dtype=tf.float32, name=None):
    """
    Compute the mel-scale spectrogram
    Args:
        spectrogram: Power spectrogram of the audio
        sample_rate: Sample rate of the audio
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel spectrum
        upper_edge_hertz: The desired top edge of the highest frequency band
        filterbank_channel_count: Number of filterbank channels
        dct_coefficient_count: Number of DCT coefficients

    Returns:
        A tensor of mel-scale spectrogram
    """
    # Check if inputs are valid.
    _validate_arguments(filterbank_channel_count, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype)

    # Spectrogram has a shape of (batch_size, M, N)
    input_length = spectrogram.shape[-1].value
    batch_size = spectrogram.shape[0].value
    center_freqs = np.zeros((filterbank_channel_count + 1,))
    mel_low = freq2mel(lower_edge_hertz)
    mel_high = freq2mel(upper_edge_hertz)
    mel_span = mel_high - mel_low
    mel_spacing = mel_span / (filterbank_channel_count + 1)
    for i in range(filterbank_channel_count + 1):
        center_freqs[i] = mel_low + (mel_spacing * (i + 1))

    # Always exclude DC; emulate HTK.
    hz_per_sbin = 0.5 * sample_rate / (input_length - 1)
    start_index = int(1.5 + (lower_edge_hertz / hz_per_sbin))
    end_index = int(upper_edge_hertz / hz_per_sbin)

    # Maps the input spectrum bin indices to filter bank channels/indices. For
    # each FFT bin, band_mapper tells us which channel this bin contributes to
    # on the right side of the triangle.  Thus this bin also contributes to the
    # left side of the next channel's triangle response.
    band_mapper = np.zeros((input_length,))
    channel = 0
    for i in range(input_length):
        melf = freq2mel(i * hz_per_sbin)
        if (i < start_index) or (i > end_index):
            band_mapper[i] = -2
        else:
            while (channel < filterbank_channel_count) and (center_freqs[channel] < melf):
                channel += 1
            band_mapper[i] = channel - 1

    # Create the weighting functions to taper the band edges.  The contribution
    # of any one FFT bin is based on its distance along the continuum between two
    # mel-channel center frequencies.  This bin contributes weights_[i] to the
    # current channel and 1-weights_[i] to the next channel.
    weights = np.zeros((input_length,))
    for i in range(input_length):
        channel = band_mapper[i]
        if (i < start_index) or (i > end_index):
            weights[i] = 0.0
        else:
            channel = int(channel)
            if (channel >= 0):
                weights[i] = (center_freqs[channel + 1] - freq2mel(i * hz_per_sbin)
                              ) / (center_freqs[channel + 1] - center_freqs[channel])
            else:
                weights[i] = (center_freqs[0] - freq2mel(i *
                                                         hz_per_sbin)) / (center_freqs[0] - mel_low)

    # Check the sum of FFT bin weights for every mel band to identify
    # situations where the mel bands are so narrow that they don't get
    # significant weight on enough (or any) FFT bins -- i.e., too many
    # mel bands have been requested for the given FFT size.
    bad_channels = []
    for c in range(filterbank_channel_count):
        band_weight_sum = 0.0
        for i in range(input_length):
            if band_mapper[i] == (c - 1):
                band_weight_sum += (1.0 - weights[i])
            elif (band_mapper[i] == c):
                band_weight_sum += weights[i]
        # The lowest mel channels have the fewest FFT bins and the lowest
        # weights sum.  But given that the target gain at the center frequency
        # is 1.0, if the total sum of weights is 0.5, we're in bad shape.
        if (band_weight_sum < 0.5):
            bad_channels.append(c)

    mapping_matrix_w = np.zeros((filterbank_channel_count, input_length))
    mapping_matrix_r = np.zeros((filterbank_channel_count, input_length))

    for i in range(start_index, end_index):
        channel = int(band_mapper[i])
        if (channel >= 0):
            mapping_matrix_w[channel][i] = 1.0
        channel += 1
        if (channel < filterbank_channel_count):
            mapping_matrix_r[channel][i] = 1.0

    # Make the mapping matrices to have a shape of (batch_size, filterbank_channel_count, N)
    # and the weights to have a shape of (batch_size, M, N)
    mapping_matrix_w = np.stack([mapping_matrix_w] * batch_size, axis=0)
    mapping_matrix_r = np.stack([mapping_matrix_r] * batch_size, axis=0)
    weights = np.stack([np.stack([weights] * batch_size, axis=0)] * spectrogram.shape[1], axis=1)

    # Compute the mel spectrum from the squared-magnitude FFT input by taking the
    # square root, then summing FFT magnitudes under triangular integration windows
    # whose widths increase with frequency.
    # Spectrogram has a shape of (batch_size, M, N)
    spec_val = tf.math.sqrt(spectrogram)
    weighted = tf.math.multiply(spec_val, weights)
    res = spec_val - weighted

    # Transpose weighted and res such that they have a shape of (batch_size, N, M)
    weighted = tf.transpose(weighted, perm=[0, 2, 1])
    res = tf.transpose(res, perm=[0, 2, 1])

    # return a Tensor of shape (batch_size, filterbank_channel_count, M)
    return tf.matmul(tf.cast(mapping_matrix_w, dtype), weighted) + tf.matmul(tf.cast(mapping_matrix_r, dtype), res)


def create_dct_matrix(dct_coefficient_count=13, filterbank_channel_count=40, dtype=tf.float32):
    """
    Compute the DCT transformation matrix
    Args:
        filterbank_channel_count: Number of filterbank channels
        dct_coefficient_count: Number of DCT coefficients

    Returns:
        DCT transformation matrix
    """
    fnorm = np.sqrt(2 / filterbank_channel_count)
    arg = np.pi / filterbank_channel_count

    arg_ = np.zeros((dct_coefficient_count, filterbank_channel_count))
    for i in range(dct_coefficient_count):
        for j in range(filterbank_channel_count):
            arg_[i][j] = i * arg * (j + 0.5)

    return fnorm * np.cos(arg_)


def compute_dct(log_mel_energies, dct_coefficient_count=13, filterbank_channel_count=40, dtype=tf.float32):
    """
    Compute the DCT of the log-magnitude of the mel-scale spectrogram
    Args:
        log_mel_energies: The log-magnitude of the mel-scale spectrogram
        filterbank_channel_count: Number of filterbank channels
        dct_coefficient_count: Number of DCT coefficients

    Returns:
        MFCC features
    """
    # log_mel_energies has shape (batch_size, filterbank_channel_count, M)
    batch_size = log_mel_energies.shape[0]

    cosines = create_dct_matrix(
        dct_coefficient_count, filterbank_channel_count)

    # Make cosines to have shape (batch_size, dct_coefficient_count, filterbank_channel_count)
    cosines = tf.cast(np.stack([cosines] * batch_size, axis=0), dtype)

    input_length = log_mel_energies.shape[1]
    if input_length > filterbank_channel_count:
        input_length = filterbank_channel_count

    # return a Tensor of shape (batch_size, N, dct_coefficient_count)
    return tf.transpose(tf.matmul(cosines[:, :input_length, :], log_mel_energies), perm=[0, 2, 1])


def compute_mfcc(spectrogram, sample_rate, lower_edge_hertz=20,
                 upper_edge_hertz=4000, filterbank_channel_count=40, dct_coefficient_count=13, dtype=tf.float32):
    """
    Compute the MFCC features
    Args:
        spectrogram: Power spectrogram of the audio
        sample_rate: Sample rate of the audio
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel spectrum
        upper_edge_hertz: The desired top edge of the highest frequency band
        filterbank_channel_count: Number of filterbank channels
        dct_coefficient_count: Number of DCT coefficients

    Returns:
        MFCC features
    """
    # Compute mfcc filterbanks.
    vals = compute_mfcc_mel_filterbank(
        spectrogram, sample_rate, lower_edge_hertz, upper_edge_hertz, filterbank_channel_count)

    # Set small values to 1e-12 so that log calculation doesn't run into trouble.
    kFilterbankFloor = 1e-12
    vals = vals * tf.cast(vals > 0, dtype) + kFilterbankFloor
    log_mel_energies = tf.math.log(vals)

    # Return results of DCT.
    return compute_dct(log_mel_energies, dct_coefficient_count)