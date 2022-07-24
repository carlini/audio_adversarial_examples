import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import pandas as pd

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

sys.path.append("DeepSpeech")
import DeepSpeech
from tf_logits import get_logits
from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.config import Config, initialize_globals
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
import absl.flags

f = absl.flags
# define parsing arguments
f.DEFINE_string('input', None, 'Input audio .wav file(s), at 16KHz (separated by spaces)')
f.DEFINE_string('restore_path', None, 'Path to the DeepSpeech checkpoint (ending in best_dev-1466475)')
f.register_validator('input',
                         os.path.isfile,
                         message='The input audio pointed to by --input must exist and be readable.')


def classify():
    with tf.Session() as sess:
        if FLAGS.input.split(".")[-1] == 'mp3':
            raw = pydub.AudioSegment.from_mp3(FLAGS.input)
            audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
        elif FLAGS.input.split(".")[-1] == 'wav':
            _, audio = wav.read(FLAGS.input)
            # for audios with 2 channels, take 2nd channel
            if (audio.shape[-1] == 2):
                audio = np.squeeze(audio[:,1])
                print(audio.shape)
        else:
            raise Exception("Unknown file format")
        
        N = audio.shape[0]
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])
        
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            # Here we should be using the preprocessing step from DS v0.9.3
            logits = get_logits(new_input, lengths)
     
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.restore_path)
        
        #  # Apply softmax for CTC decoder
        probs = tf.nn.softmax(logits, name='logits')
        probs = tf.squeeze(probs)
        
        # length was previously (N-1)//320 
        length = (N-(2*Config.audio_step_samples/3))//320
        r = sess.run(probs, {new_input: [audio],
                               lengths: [length]})
        
        if FLAGS.scorer_path:
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            scorer = None
        decoded = ctc_beam_search_decoder(r, Config.alphabet, FLAGS.beam_width,
                                          scorer=scorer, cutoff_prob=FLAGS.cutoff_prob,
                                          cutoff_top_n=FLAGS.cutoff_top_n)
        
        print("-"*80)
        print("-"*80)
        print("Classification:")
        print(decoded[0][1])
        print("-"*80)
        print("-"*80)
        
        data_dict = {'name': [FLAGS.input], 'transcript': [decoded[0][1]]}
        df = pd.DataFrame(data_dict, columns=['name', 'transcript'])
        csv_filename = "tmp/classify-{}.csv".format(time.strftime("%Y%m%d-%H%M%S"))    
        df.to_csv(csv_filename, index=False, header=True)   

def main(_):
    initialize_globals()
    classify()
        
        
def run_script():
    create_flags()
    absl.app.run(main)


if __name__ == '__main__':
    run_script()
    