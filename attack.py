## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
from shutil import copyfile
import scipy.io.wavfile as wav
import struct
import pandas as pd
import time
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech
try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.config import Config, initialize_globals
import absl.flags

# Define arguments to be parsed
f = absl.flags
f.DEFINE_multi_string('input', None, 'Input audio .wav file(s), at 16KHz (separated by spaces)')
f.DEFINE_multi_string('output', None, 'Path for the adversarial example(s)')
f.DEFINE_string('outprefix', None, 'Prefix of path for adversarial examples')
f.DEFINE_string('target', None, 'Target transcription')
f.DEFINE_multi_string('finetune', None, 'Initial .wav file(s) to use as a starting point')
f.DEFINE_integer('lr', 100, 'Learning rate for optimization')
f.DEFINE_integer('iterations', 1000, 'Maximum number of iterations of gradient descent')
f.DEFINE_float('l2penalty', float('inf'), 'Weight for l2 penalty on loss function')
f.DEFINE_boolean('mp3', False, 'Generate MP3 compression resistant adversarial examples')
f.DEFINE_string('restore_path', None, 'Path to the DeepSpeech checkpoint (ending in best_dev-1466475)')
f.DEFINE_string('lang', "en", 'Language of the input audio (English: en, German: de)')

# Define which arguments are required
f.mark_flag_as_required('input')
f.mark_flag_as_required('target')
f.mark_flag_as_required('restore_path')
    

def convert_mp3(new, lengths):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]),
                               -2**15, 2**15-1),dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])[np.newaxis,:lengths[0]]
    return mp3ed
    

class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        Set up the attack procedure.
        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        print("\nInitializing attack..\n")
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)
    
        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1) + l2penalty*ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
            
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        else:
            raise NotImplemented

        self.loss = loss
        self.ctcloss = ctcloss
        
        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Convert logits to probs for CTC decoder using softmax
        self.probs = tf.squeeze(tf.nn.softmax(self.logits, name='logits'))
        
        # Initialize scorer for CTC decoder
        if FLAGS.scorer_path:
            self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            self.scorer = None
        print("Initialization done.\n")


    def attack(self, audio, lengths, target, toks, finetune=None):
        print("Start attack..\n")
        sess = self.sess

        # Initialize all of the variables
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-(2*Config.audio_step_samples/3))//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        #now = time.time()
        MAX = self.num_iterations
        first_hits = np.zeros((self.batch_size,))
        best_hits = np.zeros((self.batch_size,))
        for i in range(MAX):
            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                new, delta, probs_out, r_logits = sess.run((self.new_input, self.delta, self.probs, self.logits))

                lst = [(probs_out, r_logits)]
                if self.mp3:
                    # TODO: Implement mp3 support 
                    raise NotImplemented("The current version does not support mp3 conversion.")
                    mp3ed = convert_mp3(new, lengths)
                    mp3_probs, mp3_logits = sess.run((self.probs, self.logits),
                                                   {self.new_input: mp3ed})
                    mp3_out = ctc_beam_search_decoder(mp3_probs, Config.alphabet, FLAGS.beam_width,
                                                scorer=self.scorer, cutoff_prob=FLAGS.cutoff_prob,
                                                cutoff_top_n=FLAGS.cutoff_top_n)
                    lst.append((mp3_out, mp3_logits))
                
                batch_size = r_logits.shape[1]
                for out, logits in lst:
                    out_list = []
                    for ii in range(batch_size):
                        if batch_size == 1:
                            probs = probs_out
                        else:
                            probs = probs_out[:,ii,:]
                        decoded = ctc_beam_search_decoder(probs, Config.alphabet, FLAGS.beam_width,
                                                        scorer=self.scorer, cutoff_prob=FLAGS.cutoff_prob,
                                                        cutoff_top_n=FLAGS.cutoff_top_n)
                        # Here we print the strings that are recognized.
                        print(decoded[0][1])
                        out_list.append(decoded)
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-int(2*Config.audio_step_samples/3))//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))


            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # Actually do the optimization step
            d, el, cl, l, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss, self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    
            # Report progress
            print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (self.loss_fn == "CTC" and i%10 == 0 and out_list[ii][0][1] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss=%f bound=%f"%(ii, cl[ii], 2000*rescale[ii][0]))
                    
                    if (first_hits[ii] == 0):
                        print("First hit for audio {} at iteration {}".format(ii, i))
                        first_hits[ii]=i
                    else:
                        best_hits[ii]=i

                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))
        
        return final_deltas, first_hits, best_hits  
    

def main(_):
    initialize_globals()
    # These are the tokens that we're allowed to use.
    # The - token is special and corresponds to the epsilon
    # value in CTC decoding, and can not occur in the phrase.
    toks = " abcdefghijklmnopqrstuvwxyz'-"
    
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []
        names = []
        source_dBs = []
        distortions = []
        high_pertub_bounds = []
        low_pertub_bounds = []

        if FLAGS.output is None:
            assert FLAGS.outprefix is not None
        else:
            assert FLAGS.outprefix is None
            assert len(FLAGS.input) == len(FLAGS.output)
        if FLAGS.finetune is not None and len(FLAGS.finetune):
            assert len(FLAGS.input) == len(FLAGS.finetune)
            
        # Load the inputs that we're given
        # TODO: [FINDBUG] loading multiple inputs is possible, 
        #       but there are some weird things going on at the end of every transcription 
        for i in range(len(FLAGS.input)):
            fs, audio = wav.read(FLAGS.input[i])
            names.append(FLAGS.input[i])
            assert fs == 16000
            assert audio.dtype == np.int16
            if (audio.shape[-1] == 2):
                audio = np.squeeze(audio[:,1])
                print(audio.shape)
            source_dB = 20 * np.log10(np.max(np.abs(audio)))
            print('source dB', source_dB)
            source_dBs.append(source_dB)
            audios.append(list(audio))
            lengths.append(len(audio))

            if FLAGS.finetune is not None:
                finetune.append(list(wav.read(FLAGS.finetune[i])[1]))   
            
        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        
        phrase = FLAGS.target 
        print("\nAttack phrase: ", phrase) 
        
        attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=len(audios),
                        mp3=FLAGS.mp3,
                        learning_rate=FLAGS.lr,
                        num_iterations=FLAGS.iterations,
                        l2penalty=FLAGS.l2penalty,
                        restore_path=FLAGS.restore_path)

        start_time = time.time() 
        deltas, first_hits, best_hits = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               toks,
                               finetune)
        runtime = time.time() - start_time

        print("Finished in {}s.".format(runtime))
        # And now save it to the desired output
        if FLAGS.mp3:
            convert_mp3(deltas, lengths)
            copyfile("/tmp/saved.mp3", FLAGS.output[0])
            print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]]-audios[0][:lengths[0]])))
        else:
            for i in range(len(FLAGS.input)):
                if FLAGS.output is not None:
                    path = FLAGS.output[i]
                else:
                    path = FLAGS.outprefix+str(i)+".wav"
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
                
                # Define metrics for evaluation
                diff = deltas[i][:lengths[i]]-audios[i][:lengths[i]]
                high_pertub_bound = np.max(np.abs(diff))
                low_pertub_bound = np.min(np.abs(diff[diff!=0]))
                distortion = 20 * np.log10(np.max(np.abs(diff))) - source_dBs[i]
                high_pertub_bounds.append(high_pertub_bound)
                low_pertub_bounds.append(low_pertub_bound)
                distortions.append(distortion)
                print("Final noise loudness: ", distortion)

    # Create data_dict to store values for csv file
    data_dict = {
        'filename': names,
        'length' : lengths,
        'attack_runtime': [runtime]*len(names),
        'source_dB': source_dBs,
        'noise_loudness': distortions,
        'high_pertubation_bound' : high_pertub_bounds,
        'low_pertubation_bound' : low_pertub_bounds,
        'first_hit' : first_hits,
        'best_hit' : best_hits
    }     
    df = pd.DataFrame(data_dict, columns=['filename', 'length', 'attack_runtime', 'source_dB', 'noise_loudness', 'high_pertubation_bound', 'low_pertubation_bound', 'first_hit', 'best_hit'])
    csv_filename = "tmp/attack-{}.csv".format(FLAGS.lang, time.strftime("%Y%m%d-%H%M%S"))    
    df.to_csv(csv_filename, index=False, header=True)   
 
                
def run_script():
    create_flags()
    absl.app.run(main)
    
    
if __name__ == "__main__":
    run_script()