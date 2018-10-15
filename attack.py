## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None


from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

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
                 mp3=False, l2penalty=float('inf')):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
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
        saver.restore(sess, "models/session_dump")

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths, batch_size)
            
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
            raise

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

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, audio, lengths, target, finetune=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
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
        now = time.time()
        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            now = time.time()

            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]
                if self.mp3:
                    mp3ed = convert_mp3(new, lengths)
                    
                    mp3_out, mp3_logits = sess.run((self.decoded, self.logits),
                                                   {self.new_input: mp3ed})
                    lst.append((mp3_out, mp3_logits))

                for out, logits in lst:
                    chars = out[0].values

                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))


            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # Actually do the optimization ste
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
                if (self.loss_fn == "CTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
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

                    print("Worked i=%d ctcloss=%f bound=%f"%(ii,cl[ii], 2000*rescale[ii][0]))
                    #print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))

        return final_deltas


    
def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")
    parser.add_argument('--out', type=str, nargs='+',
                        required=False,
                        help="Path for the adversarial example(s)")
    parser.add_argument('--outprefix', type=str,
                        required=False,
                        help="Prefix of path for adversarial examples")
    parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
    parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate for optimization")
    parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
    parser.add_argument('--mp3', action="store_const", const=True,
                        required=False,
                        help="Generate MP3 compression resistant adversarial examples")
    args = parser.parse_args()
    
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []

        if args.out is None:
            assert args.outprefix is not None
        else:
            assert args.outprefix is None
            assert len(args.input) == len(args.out)
        if args.finetune is not None and len(args.finetune):
            assert len(args.input) == len(args.finetune)
        
        # Load the inputs that we're given
        for i in range(len(args.input)):
            fs, audio = wav.read(args.input[i])
            assert fs == 16000
            assert audio.dtype == np.int16
            print('source dB', 20*np.log10(np.max(np.abs(audio))))
            audios.append(list(audio))
            lengths.append(len(audio))

            if args.finetune is not None:
                finetune.append(list(wav.read(args.finetune[i])[1]))

        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

        phrase = args.target

        # Set up the attack class and run it
        attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=len(audios),
                        mp3=args.mp3,
                        learning_rate=args.lr,
                        num_iterations=args.iterations,
                        l2penalty=args.l2penalty)
        deltas = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               finetune)

        # And now save it to the desired output
        if args.mp3:
            convert_mp3(deltas, lengths)
            copyfile("/tmp/saved.mp3", args.out[0])
            print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]]-audios[0][:lengths[0]])))
        else:
            for i in range(len(args.input)):
                if args.out is not None:
                    path = args.out[i]
                else:
                    path = args.outprefix+str(i)+".wav"
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
                print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]]-audios[i][:lengths[i]])))

main()
