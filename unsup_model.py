# 2016 Benjamin Milde, TU-Darmstadt
# 2017 Benjamin Milde, Universitaet Hamburg
#
# Inspired by https://github.com/dennybritz/cnn-text-classification-tf, as it also uses 1-D convolutions
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import utils
import math
from tensorflow.python.platform import gfile
import matplotlib
import matplotlib.pyplot as pyplot
from experimental_rnn.rnn_cell_mulint_modern import HighwayRNNCell_MulInt, GRUCell_MulInt
import cwrnn

tf.flags.DEFINE_string("filter_sizes", "200", "Comma-separated filter sizes (default: '3,4,5')") # 25ms @ 16kHz
tf.flags.DEFINE_integer("num_filters", 40, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_integer("window_length", 1600, "Window length") # 100 ms @ 16kHz
tf.flags.DEFINE_integer("output_length", 200, "Output length") # ~12 ms @ 16kHz

tf.flags.DEFINE_integer("fc_size", 128 , "Fully connected size at the end of the network.")
tf.flags.DEFINE_integer("decoder_layers", 5 , "Decoder layers.")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5 , "Dropout keep probability")

# Training parameters
tf.flags.DEFINE_string("filelist", "TEDLIUM2_wav.txt", "Filelist, one wav file per line")
tf.flags.DEFINE_string("cost_function", "mse", "Type of loss function to use for the model. Can be mse, mase, deriv, e_mse, e_mse_deriv.")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

tf.flags.DEFINE_integer("steps_per_checkpoint", 500,
                                "How many training steps to do per checkpoint.")
tf.flags.DEFINE_integer("steps_per_summary", 100,
                                "How many training steps to do per checkpoint.")

tf.flags.DEFINE_integer("checkpoints_per_save", 1,
                                "How many checkpoints until saving the model.")

#tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


tf.flags.DEFINE_boolean("eval", False, "Eval instead of training")
tf.flags.DEFINE_boolean("show_feat", False, "Display features of the encoder")
tf.flags.DEFINE_float("temp", 0.8,"Temperature for sampling")
tf.flags.DEFINE_integer("gen_steps", 500,"How many (full) prediction steps to do for the generation.")

tf.flags.DEFINE_boolean("debug", False, "E.g. Smaller training data size")

tf.app.flags.DEFINE_boolean("log_tensorboard", True, "Log training process if this is set to True.")

# Model dir
tf.flags.DEFINE_string("train_dir", "", "Training dir to resume training from. If empty, a new one will be created.")

#
FLAGS = tf.flags.FLAGS

training_data = {}

#from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#compresses the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def encode_mulaw(signal,mu=255):
    return np.sign(signal)*(np.log(1.0+mu*np.abs(signal)) / np.log(1.0+mu))

#uncompress the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def decode_mulaw(signal,mu=255):
    return np.sign(signal)*(1.0/mu)*(np.power(1+mu,np.abs(signal))-1.0)

def get_FLAGS_params_as_str():
    params_str = ''
    for attr, value in sorted(FLAGS.__flags.items()):
        params_str += "{}={}\n".format(attr.upper(), value)
    return params_str

# discretize signal between -1.0 and 1.0 into mu+1 bands.
def discretize(signal, mu=255.0):
    output = np.array(signal)
    output += 1.0
    output = output*(0.5*mu)
    signal = np.fmax(0.0,output)
    #signal = np.fmin(255.0,signal)
    return signal.astype(np.int32)

def undiscretize(signal, mu=255.0):
    output = np.array(signal)
    output = output.astype(np.float32)
    output /= 0.5*mu
    output -= 1.0
    signal = np.fmax(-1.0,output)
    signal = np.fmin(1.0,signal)
    return signal

class UnsupSeech(object):
    """
    Unsupervised learning with RAW speech signals
    """

    def maxpool1d(self, input_tensor, temporal_pool=1, channel_pool=1):
        return tf.nn.max_pool(
                    input_tensor,
                    ksize=[1, 1, 2, 1],
                    strides=[1, 1, 2, 1],
                    padding='VALID',
                    name="pool")

    def fully_connected(self, in_tensor, in_size, out_size, name='fc', non_linearity=tf.nn.relu, use_dropout=True, dropout_keep_prob=0.8):
	    with tf.variable_scope(name):
			wd = tf.get_variable('wd', shape=[in_size, out_size],initializer=tf.contrib.layers.xavier_initializer())
			bd = tf.get_variable('bd', shape=[out_size], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(out_size)))

			print('Fully connected layer: '+ name)
			print('in_tensor shape:',in_tensor.get_shape())
			print('bd shape:',bd.get_shape())
			print('wd shape:',wd.get_shape())

			out_tensor = tf.add(tf.matmul(in_tensor, wd), bd)
            if non_linearity is not None:
                out_tensor = non_linearity(out_tensor)
            if use_dropout:
                out_tensor = tf.nn.dropout(out_tensor, dropout_keep_prob)
            return out_tensor

    def create_training_graphs(self, create_new_train_dir=True, clip_norm=True, max_grad_norm=0.5):
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-4)

        if clip_norm:
            tvars = tf.trainable_variables()
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)
            self.grads_and_vars = zip(self.grads, tvars)
        else:
            self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
        
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        #self.train_op = self.optimizer.minimize(self.cost)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        if create_new_train_dir:
            timestamp = str(int(time.time()))
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp)) + '/'
            print("Writing to {}\n".format(self.out_dir))
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            with open(self.out_dir + 'params','w') as param_file:
                param_file.write(get_FLAGS_params_as_str()+'\n')
        else:
            self.out_dir = FLAGS.train_dir 

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", self.cost)
        #acc_summary = tf.scalar_summary("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")

        # Dev summaries
        #dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #self.dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        self.saver = tf.train.Saver(tf.all_variables()) 

    # get a batch
    def get_batch(self, filelist, input_size, output_size, batch_size, model = 'post_pred'):
        
        input_x = []
        input_y = []
        decoder_inputs = []

        filelist_size = len(filelist)
        
        for i in xrange(batch_size):
            random_file_num = int(math.floor(np.random.random_sample() * filelist_size))
            random_file = filelist[random_file_num]
        
            audio_data = training_data[random_file] 
            #print('audio data shape:',audio_data.shape)
            audio_len = audio_data.shape[0] - output_size - input_size

            random_pos_num = int(math.floor(np.random.random_sample() * audio_len))
            
            input_slice = audio_data[random_pos_num:random_pos_num+input_size]
            output_slice = audio_data[random_pos_num+input_size-1:random_pos_num+input_size+output_size]

            output_slice_dis = discretize(output_slice)
            #print(input_slice, output_slice)

            input_x.append(input_slice)
            input_y.append(output_slice_dis[1:])
            decoder_inputs.append(output_slice_dis[:-1])

        return input_x,input_y,decoder_inputs

    def __init__(self, window_length, output_length, filter_sizes, num_filters, fc_size, dropout_keep_prob, train_files, is_training=True, cost_function='mse', create_new_train_dir=True, mu=255, emb_size=4, batch_size=128, decoder_layers=3):

        self.train_files = train_files

        self.output_length = output_length
        self.fc_size = fc_size
        self.window_length = window_length

        # None -> automatically set to batch_size
        # window length 80 sample = 5ms at 16kHz
        self.input_x = tf.placeholder(tf.float32, [None, window_length], name="input_x")
        
        self.decoder_inputs = tf.placeholder(tf.int32, [None, output_length], name="decoder_inputs")
        self.input_symbol = tf.placeholder(tf.int32, [None, 1])
        self.input_y = tf.placeholder(tf.int32, [None, output_length], name="input_y")
        self.input_state = tf.placeholder(tf.float32, [None, fc_size*decoder_layers])
        
        with tf.name_scope("unsupmodel"):
            input_reshaped = tf.reshape(self.input_x, [-1, 1, window_length, 1])

            print('input_shape:', input_reshaped)

            self.pooled_outputs = []

            #currently we only support one filtersize (but we could extend)
            #for i, filter_size in enumerate(filter_sizes):
            filter_size = filter_sizes[0]
            i=0

            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [1 , filter_size, 1, num_filters]
                print('filter_shape:',filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                #W = tf.get_variable("W",shape=filter_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
                
                # tf.nn.atrous_conv2d?

                # 1D conv without padding(padding=VALID)
                conv = tf.nn.conv2d(input_reshaped,W,strides=[1, 1, 2, 1],padding="VALID",name="conv")

                ## Apply nonlinearity
                b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b1")
                conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")

                pool_input_dim = conv.get_shape()[2]

                print('conv1 shape:',conv.get_shape())
                # Temporal maxpool accross all filters, pool size 2
                pooled = tf.nn.max_pool(conv,ksize=[1, 1, pool_input_dim / 8, 1], # max_pool over / 4 of inputsize filters
                                        strides=[1, 1, pool_input_dim / 16 , 1], # hopped by / 8 of input size
                                        padding='VALID',name="pool")

                print('pool1 shape:',pooled.get_shape())

                #input shape: batch, in_height, in_width, in_channels
                #filter shape: filter_height, filter_width, in_channels, out_channels
                #('pool1 shape:', TensorShape([Dimension(None), Dimension(1), Dimension(7), Dimension(80)]))

                if second_cnn_layer:
                    filter_shape = [1, 7, num_filters, num_filters*4]
                    print('filter_shape conv2:',filter_shape)
                    W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W2")

                    #b2 = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b2")

                    conv = tf.nn.conv2d(pooled, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv")

                    ## Apply nonlinearity
                    b = tf.Variable(tf.constant(0.01, shape=[filter_shape[-1]]), name="b2")
                    conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")

                    pool_input_dim = conv.get_shape()[2]
                    print('conv2 shape:',conv.get_shape())

                    pooled = tf.nn.max_pool(conv,ksize=[1, 1, pool_input_dim, 1], # pool over all outputs from previous layer
                                            strides=[1, 1, 1 , 1], # no stride
                                            padding='VALID',name="pool")

                    print('pool2 shape:',pooled.get_shape())

                #self.pooled_outputs.append(pooled)

                flattened_size = int(pooled.get_shape()[2]*pooled.get_shape()[3])
                # Reshape conv2 output to fit fully connected layer input
                self.flattened_pooled = tf.reshape(pooled, [-1, flattened_size])
			
                print('flattened_pooled shape:',self.flattened_pooled.get_shape())

                self.fc1 = self.fully_connected(self.flattened_pooled, flattened_size, fc_size*decoder_layers, name='fc1', use_dropout=is_training)
                self.fc2 = self.fully_connected(self.fc1, fc_size, fc_size, name='fc2', use_dropout=is_training)
                
                #single_cell = tf.nn.rnn_cell.GRUCell(fc_size)
                #single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=0.8, state_is_tuple=False)
                print('Decoder size: %d, layers %d' % (fc_size , decoder_layers))
                single_cell = GRUCell_MulInt(fc_size, use_recurrent_dropout=is_training, recurrent_dropout_factor = 0.9 if is_training else 1.0)
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * decoder_layers, state_is_tuple=False)
                
                cell = cwrnn.CWRNNCell([single_cell] * decoder_layers, [1,4,8,16,32,64,128,256,512,1024][:decoder_layers], state_is_tuple=False)
                #single_cell = HighwayRNNCell_MulInt(fc_size, num_highway_layers=decoder_layers, 
                #                use_recurrent_dropout=is_training, recurrent_dropout_factor=0.9)
                #cell = single_cell

                state = self.fc2
                self.initial_state = self.fc2                

                embedding = tf.get_variable("embedding", [mu+1, emb_size], initializer=tf.random_uniform_initializer(-1,1))
                self.decoder_inputs_emb = tf.nn.embedding_lookup(embedding, self.decoder_inputs)
                #self.decoder_first_input_emb = self.decoder_inputs_emb[:,0,:]
                self.input_symbol_emb = tf.nn.embedding_lookup(embedding, self.input_symbol)

                #next_input_emb = self.decoder_first_input_emb

                softmax_w = tf.get_variable("softmax_w", shape=[fc_size*decoder_layers, mu], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", shape=[mu], dtype=tf.float32)
                
                rnn_outputs = []
                #for training: multiple RNN steps
                with tf.variable_scope("decoderRNN"):
                    for time_step in range(output_length):
                        if time_step > 0: tf.get_variable_scope().reuse_variables()
                        #print('decoder_inputs shape:',self.decoder_inputs_emb[:,time_step,:].get_shape())
                        #print('state shape:',state.get_shape())
                        next_input_emb = self.decoder_inputs_emb[:,time_step,:]
                        (cell_output, state) = cell(next_input_emb, state)
                        logits = tf.matmul(cell_output, softmax_w) + softmax_b        
                        rnn_outputs.append(logits)
                        #print('logits shape:',logits.get_shape())
                        #output_symbols = tf.argmax(tf.nn.softmax(logits),1)
                        #print('output_symbols shape:', output_symbols.get_shape())
                        #next_input_emb = tf.nn.embedding_lookup(embedding, output_symbols)
                        #print('next_input shape:', next_input_emb.get_shape())

               #[self.cell_output_softmax, self.output_state]
               
               ##single RNN step
                with tf.variable_scope("decoderRNN"):
                    tf.get_variable_scope().reuse_variables()
                    (rnn_step_cell_output, rnn_step_state) = cell(self.input_symbol_emb[:,0,:], self.input_state)
                    self.cell_output = rnn_step_cell_output
                    self.output_state = rnn_step_state
                    self.cell_output_logits = tf.matmul(self.cell_output, softmax_w) + softmax_b
                    self.cell_output_softmax = tf.nn.softmax(self.cell_output_logits)

                rnn_output = tf.reshape(tf.concat(1, rnn_outputs), [-1, mu])
                print('rnn_output shape:',rnn_output.get_shape())
                self.out = tf.reshape(tf.argmax(rnn_output,1),[-1, output_length])
                print('out shape (argmaxes):',self.out.get_shape())

#               self.out = self.fully_connected(self.fc2, fc_size, output_length, name='out', non_linearity=None, use_dropout=False)

                #loss = tf.nn.seq2seq.sequence_loss_by_example(rnn_outputs,self.input_y,
                #                    tf.ones([batch_size, output_length], dtype=tf.float32))
                
                self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rnn_output, tf.reshape(self.input_y,[-1]))) / batch_size
                
                #self.cost = tf.reduce_sum(loss) / batch_size

                # Compressive non-linearity (Speech acoustic modeling from raw multichannel waveforms)
                #self.fc1 = tf.log(self.fc1 + 0.01)

                #self.energy_cost_reduced = tf.abs(tf.reduce_mean(tf.abs(self.input_y)) - tf.reduce_mean(tf.abs(self.out)) )
                #energy_cost_factor = 0.15

                # minimize squared error
                #self.cost = tf.reduce_sum(tf.pow((self.input_y+1.0) - (self.out+1.0), 2) +
                #        tf.pow((self.input_y-1.0) - (self.out-1.0), 2)) + self.energy_cost_reduced * energy_cost_factor

                #cost_function = 'mse'
                #self.cost = tf.reduce_mean(tf.pow((self.input_y) - (self.out), 2) + 1.0) + self.energy_cost_reduced
                #if cost_function == 'mase':
                #    self.cost = tf.reduce_mean(tf.abs(self.input_y - self.out)) #+ energy_cost_factor*self.energy_cost_reduced
                #elif cost_function == 'mse':
                #    self.cost = tf.reduce_mean(tf.pow(self.input_y - self.out, 2))
                #elif cost_function == 'e_mse':
               #     self.cost = tf.reduce_mean(tf.pow(self.input_y - self.out, 2)) + energy_cost_factor*self.energy_cost_reduced
               # elif cost_function == 'deriv':
               #     paddings_left = [[0,0],[1,0]]
               #     paddings_right = [[0,0],[0,1]]

               #     deriv_out = tf.pad(self.out, paddings_left, "CONSTANT") - tf.pad(self.out, paddings_right, "CONSTANT")
               #     deriv_input_y = tf.pad(self.input_y, paddings_left, "CONSTANT") - tf.pad(self.input_y, paddings_right, "CONSTANT")

               #    self.cost = 1.0*tf.reduce_mean(tf.pow(deriv_out - deriv_input_y, 2))

               # elif cost_function == 'mse_deriv':
               #     paddings_left = [[0,0],[1,0]]
               #     paddings_right = [[0,0],[0,1]]

               #     deriv_out = tf.pad(self.out, paddings_left, "CONSTANT") - tf.pad(self.out, paddings_right, "CONSTANT")
               #     deriv_input_y = tf.pad(self.input_y, paddings_left, "CONSTANT") - tf.pad(self.input_y, paddings_right, "CONSTANT")

               #     self.cost = 5.0*tf.reduce_mean(tf.pow(deriv_out - deriv_input_y, 2)) + 1.0*tf.reduce_mean(tf.pow(self.input_y - self.out, 2))

               # elif cost_function == 'e_mse_deriv':
               #     paddings_left = [[0,0],[1,0]]
               #     paddings_right = [[0,0],[0,1]]

               #     deriv_out = tf.pad(self.out, paddings_left, "CONSTANT") - tf.pad(self.out, paddings_right, "CONSTANT")
               #     deriv_input_y = tf.pad(self.input_y, paddings_left, "CONSTANT") - tf.pad(self.input_y, paddings_right, "CONSTANT")
#
               #     self.cost = 5.0*tf.reduce_mean(tf.pow(deriv_out - deriv_input_y, 2)) 
               #     + 1.0*tf.reduce_mean(tf.pow(self.input_y - self.out, 2)) + 2.0*self.energy_cost_reduced

            self.create_training_graphs(create_new_train_dir)
            self.nan_checker = tf.add_check_numerics_ops()

    def run_rnn_step(self, sess, input_symbol, input_state, batch_size=1):
        data = np.zeros([batch_size, 1], dtype=np.int32)
        for i,symbol in enumerate(input_symbol):
            data[i] = symbol

        return sess.run([self.cell_output_softmax, self.output_state],{self.input_state: input_state,
									  self.input_symbol: data})

    # do a training step with the supplied input data
    def step(self, sess, input_x, input_y, decoder_inputs):
        feed_dict = {self.input_x: input_x, self.input_y: input_y, self.decoder_inputs: decoder_inputs}
        _, output, loss = sess.run([self.train_op, self.out, self.cost], feed_dict=feed_dict)
        return  output, loss

    # generate features for 1D numpy input vector (audio in time domain)
    def gen_feat(self, sess, np_signal):
        feed_dict = {self.input_x: [np_signal]}
        feat = sess.run(self.initial_state, feed_dict=feed_dict)
        return feat[0]

    # generate features for 1D numpy input vector (audio in time domain)
    # batched version
    def gen_feat_batch(self, sess, np_signals):
        feed_dict = {self.input_x: np_signals}
        feats = sess.run(self.flattened_pooled, feed_dict=feed_dict)
        return feats

    #def gen_next_batch(self, sess, np_signals):
    #    feed_dict = {self.input_x: np_signals}
    #    signals = sess.run(self.out, feed_dict=feed_dict)
    #    return signals

    def generate_signal(self, sess, np_signal, temperature=1.0):
        feed_dict = {self.input_x: [np_signal]}
        state = [self.gen_feat(sess, np_signal)]
        print ("np_signal[-1]: ", np_signal[-1])
        print state
        input_symbol = discretize([np_signal[-1]])[0]
        print("input_symbol: %d" % input_symbol)
        generated = []
        for i in xrange(self.output_length):
            cell_output,state = self.run_rnn_step(sess, [input_symbol], state)
            #input_symbol = np.argmax(cell_output[0])
            input_symbol = sample(cell_output[0],temperature=temperature)
            generated.append(input_symbol)
        print("Generated:",generated)
        return undiscretize(generated)

def gen_feat(filelist, sample_data=True):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/cpu:0'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_length=FLAGS.window_length, output_length=FLAGS.output_length, filter_sizes=filter_sizes,
                                num_filters=FLAGS.num_filters, fc_size=FLAGS.fc_size, dropout_keep_prob=1.0, train_files = filelist, create_new_train_dir = False, is_training=False, decoder_layers=FLAGS.decoder_layers)
            if FLAGS.train_dir != "":
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    # model is now loaded with the trained parameters
                    for myfile in filelist:
                        input_signal = training_data[myfile][20*16000:]
                        if FLAGS.show_feat:
                            feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, 180)[:500])
                            pyplot.imshow(feat.T)
                            pyplot.show()
                            print feat
                        pre_sig_length = 1600 #1450
                        gen_signal = input_signal[:pre_sig_length]
                        print 'Generating signal...'
                        for i in xrange(FLAGS.gen_steps):
                            next_signal = model.generate_signal(sess, gen_signal[-FLAGS.window_length:], temperature=FLAGS.temp)
                            #model.gen_next_batch(sess, [gen_signal[-FLAGS.window_length:]])
                            #input_signal = input_signal[FLAGS.output_length:] + next_signal[0]
                            if i % 100 == 0:
                                print next_signal[0]
                                print(gen_signal.shape)
                            gen_signal = np.append(gen_signal,next_signal)
                        utils.writeSignal(gen_signal,'gentest.wav')
                        print 'done!'
                else:
                    print("Could not open training dir: %s" % FLAGS.train_dir)
            else:
                print("Train_dir parameter is empty")

def train(filelist):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/gpu:1'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_length=FLAGS.window_length, output_length=FLAGS.output_length, filter_sizes=filter_sizes, 
                                num_filters=FLAGS.num_filters, fc_size=FLAGS.fc_size, dropout_keep_prob=FLAGS.dropout_keep_prob, train_files = filelist, cost_function=FLAGS.cost_function, batch_size=FLAGS.batch_size, decoder_layers=FLAGS.decoder_layers)
            
            restored = False
            if FLAGS.train_dir != "":
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(session, ckpt.model_checkpoint_path)
                    restored = True
                else:
                    print("Could load parameters from" + FLAGS.train_dir)
            if not restored:
                print("Created model with fresh parameters.")
                sess.run(tf.initialize_all_variables())

            summary_writer = None
            if FLAGS.log_tensorboard:
                summary_writer = tf.train.SummaryWriter(model.out_dir, sess.graph)

            train_losses = []
            energy_losses = []

            step_time, loss = 0.0, 0.0
            current_step = 0
            checkpoint_step = 0
            previous_losses = []
            input_x, input_y = None, None

            while True:
                current_step += 1

                if current_step % FLAGS.steps_per_summary == 0 and summary_writer is not None:
                    input_x, input_y, decoder_inputs = model.get_batch(filelist=filelist, input_size=FLAGS.window_length, output_size=FLAGS.output_length, batch_size=FLAGS.batch_size, model = 'post_pred')
                    summary_str = sess.run(model.train_summary_op, feed_dict={model.input_x:input_x, model.input_y:input_y, model.decoder_inputs: decoder_inputs})
                    summary_writer.add_summary(summary_str, current_step)

                # Get a batch and make a step.
                start_time = time.time()
                input_x, input_y, decoder_inputs = model.get_batch(filelist=filelist, input_size=FLAGS.window_length, output_size=FLAGS.output_length, batch_size=FLAGS.batch_size, model = 'post_pred')
                output_y, train_loss = model.step(sess, input_x, input_y, decoder_inputs)
                #train_loss = np.log(train_loss)
                train_losses.append(train_loss)

                #energy_loss = np.log(energy_loss)
                #energy_losses.append(energy_loss)

                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    checkpoint_step += 1
                    mean_train_loss = np.mean(train_losses)
                    #mean_energy_loss = np.mean(energy_loss)

                    print('real_y:', input_y[0])
                    print('output_y:', output_y[0])
                    print('At step %i step-time %.4f loss %.4f' % (current_step, step_time, mean_train_loss))
                    #print('Input_x, input_y:', input_x[0], input_y[0])
                    train_losses = []
                    energy_losses = []
                    step_time = 0
                    if checkpoint_step % FLAGS.checkpoints_per_save == 0:
                        min_loss = 1e10
                        if len(previous_losses) > 0:
                            min_loss = min(previous_losses)
                        if mean_train_loss < min_loss:
                            print(('Train loss: %.6f' % mean_train_loss) + (' is smaller than previous best loss: %.6f' % min_loss) )
                            print('Saving the best model so far...')
                            model.saver.save(sess, model.out_dir, global_step=model.global_step)
                            previous_losses.append(mean_train_loss)


if __name__ == "__main__":
    FLAGS._parse_flags()
    print("\nParameters:")
    print(get_FLAGS_params_as_str())
    filelist = utils.loadIdFile(FLAGS.filelist, 300)
    print filelist

    if FLAGS.eval:
        filelist = utils.loadIdFile(FLAGS.filelist, 10)
        filelist = filelist[-5:]
    elif FLAGS.debug:
        filelist = filelist[:10]

    for myfile in filelist:
#    for myfile in [filelist[-1]]:   
        print('Loading:',myfile)
        signal = np.float32(utils.getSignal(myfile)[0])
        #convert and clip to -1.0 - 1.0 range
        signal /= 32768.0
        signal = np.fmax(-1.0,signal)
        signal = np.fmin(1.0,signal)
        #compress dynamic range
        signal = encode_mulaw(signal)
        #signal /= np.std(signal)
        #signal = (signal-np.mean(signal))/np.std(signal);
        training_data[myfile] = signal

    if FLAGS.eval:
        print gen_feat([filelist[-1]]) 
    else:
        train(filelist)
