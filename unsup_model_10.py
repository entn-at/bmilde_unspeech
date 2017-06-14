# Copyright 2016 Benjamin Milde, TU-Darmstadt
# Copyright 2017 Benjamin Milde, Universiaet Hamburg
#
# Inspired by https://github.com/dennybritz/cnn-text-classification-tf, as it also uses 1-D convolutions
# and Wavenet, for its idea of sampling audio as a discrete distribution
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
import sys
import time
import datetime
from tensorflow.contrib import learn
import utils
import math
from tensorflow.python.platform import gfile
import matplotlib
import matplotlib.pyplot as pyplot
from experimental_rnn.rnn_cell_mulint_modern import HighwayRNNCell_MulInt, GRUCell_MulInt
import cwrnn_10 as cwrnn
import pyplot

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

if sys.version_info[0] == 3:
    xrange = range
    
tf.flags.DEFINE_string("filter_sizes", "512", "Comma-separated filter sizes (default: '200')") # 25ms @ 16kHz
tf.flags.DEFINE_integer("num_filters", 40, "Number of filters per filter size (default: 40)")

tf.flags.DEFINE_integer("window_length", 768, "Window length") # 100+ ms @ 16kHz
tf.flags.DEFINE_integer("output_length", 16, "Output length") # 50 ms @ 16kHz

tf.flags.DEFINE_integer("fc_size", 256 , "Fully connected size at the end of the network.")
tf.flags.DEFINE_integer("decoder_layers", 2 , "Decoder layers.")

tf.flags.DEFINE_float("dropout_keep_prob", 1.0 , "Dropout keep probability")

tf.flags.DEFINE_string("decoder_type", "rnn", "Currently supported: decoder type rnn or decoder type nn (only for an output size of 1)")

tf.flags.DEFINE_boolean("decoder_state_add_initial", False, "Adds the initial state of the decoder (the representation the encoder produces) to each decoder step")
tf.flags.DEFINE_boolean("use_scheduld_sampling", False, "Wether to use the scheduld sampling strategy.")

tf.flags.DEFINE_boolean("batch_normalization", True, "Wether to use batch normalization.")

# Training parameters
tf.flags.DEFINE_string("filelist", "filelist.track1.english", "Filelist, one wav file per line")
tf.flags.DEFINE_string("cost_function", "mse", "Type of loss function to use for the model. Can be mse, mase, deriv, e_mse, e_mse_deriv.")

tf.flags.DEFINE_float("l2_regularization_strength", None, "L2 regularization strength for training, default: no regularization")
tf.flags.DEFINE_float("learn_rate", 5e-4, "Learn rate for the optimizer")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

tf.flags.DEFINE_integer("steps_per_checkpoint", 1000,
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
tf.flags.DEFINE_float("temp", 1.0,"Temperature for sampling")
tf.flags.DEFINE_integer("gen_steps", 32000,"How many (full) prediction steps to do for the generation.")

tf.flags.DEFINE_boolean("debug", False, "E.g. Smaller training data size")

tf.app.flags.DEFINE_boolean("log_tensorboard", True, "Log training process if this is set to True.")


# Model dir
tf.flags.DEFINE_string("train_dir", "/srv/data/milde/unspeech_models/", "Training dir to resume training from. If empty, a new one will be created.")
tf.flags.DEFINE_string("genwav_dir", "gentest/", "When sampling, generate wav files in this directory..")

#
FLAGS = tf.flags.FLAGS

training_data = {}

#from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.p
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
    return np.sign(signal)*(np.log1p(mu*np.abs(signal)) / np.log1p(mu))

#uncompress the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def decode_mulaw(signal,mu=255):
    return np.sign(signal)*(1.0/mu)*(np.power(1.0+mu,np.abs(signal))-1.0)

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

def pool1d(value, ksize, strides, padding, data_format="NHWC", name=None):
    """Performs the max pooling on an input with one spatial dimension.

    Args:
      value: A 3-D `Tensor` with shape `[batch, width, channels]` and
        type `tf.float32`.
      ksize: A list of ints that has length 3.  The size of the window for
        each dimension of the input tensor.
      strides: A list of ints that has length 3.  The stride of the sliding
        window for each dimension of the input tensor.
      padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
        See the @{tf.nn.convolution$comment here}
      data_format: A string. 'NHWC' and 'NCHW' are supported.
      name: Optional name for the operation.

    Returns:
      A `Tensor` with type `tf.float32`.  The max pooled output tensor.
    """
    value_rsh = tf.reshape(value, [-1, 1, int(value.shape[1]), int(value.shape[2])])
    ksize_rsh = [ksize[0], 1,  ksize[1], ksize[2]]
    strides_rsh = [strides[0], 1, strides[1], strides[2]]

    pooled = tf.nn.max_pool(value_rsh, ksize_rsh, strides_rsh, padding, data_format, name)
    result = tf.reshape(pooled, [-1, int(pooled.shape[2]), int(pooled.shape[3])])
    return result

#leakly relu to circumvent the dieing ReLU problem
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def tensor_normalize_0_to_1(in_tensor):
    x_min = tf.reduce_min(in_tensor)
    x_max = tf.reduce_max(in_tensor)
    tensor_0_to_1 = ((in_tensor - x_min) / (x_max - x_min))
    return tensor_0_to_1

#https://gist.github.com/awjuliani/fb10d1ea206fab25f946512d959e3894
def DenseBlock2D(input_layer,filters, layer_num, num_connected, non_linearity=lrelu):
    with tf.variable_scope("dense_unit"+str(layer_num)):
        nodes = []
        a = slim.conv2d(input_layer,filters,[3,3], activation_fn=non_linearity)
        nodes.append(a)
        for z in range(num_connected):
            b = slim.conv2d(tf.concat(nodes,3),filters,[3,3], activation_fn=non_linearity)
            nodes.append(b)
        return b

#https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py
def DenseTransition2D(l, filters, name, with_conv=True, non_linearity=lrelu):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name):
        if with_conv:
            l = slim.conv2d(l,filters,[3,3], activation_fn=non_linearity)
        l = slim.avg_pool2d(l, [2,2])
    #with tf.variable_scope(name) as scope:
    #   l = BatchNorm('bn1', l)
#       l = lrelu(l)
#       l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=non_linearity)
#       l = AvgPooling('pool', l, 2)
    return l

def DenseFinal2D(l, name, pool_size=7):
    with tf.variable_scope(name):
        l = slim.avg_pool2d(l, [pool_size,pool_size], stride=1)
    return l

class UnsupSeech(object):
    """
    Unsupervised learning with RAW speech signals
    """

    #def maxpool1d(self, input_tensor, temporal_pool=1, channel_pool=1):
    #    return tf.nn.max_pool(
    #                input_tensor,
    #                ksize=[1, 1, 2, 1],
    #                strides=[1, 1, 2, 1],
    #                padding='VALID',
    #                name="pool")

    def fully_connected(self, in_tensor, in_size, out_size, name='fc', non_linearity=lrelu, use_dropout=True, dropout_keep_prob=0.8):
        with tf.variable_scope(name):
            wd = tf.get_variable('w_d', shape=[in_size, out_size],initializer=tf.contrib.layers.xavier_initializer())
            bd = tf.get_variable('bias_d', shape=[out_size], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(out_size)))

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

    def create_training_graphs(self, create_new_train_dir=True, clip_norm=True, max_grad_norm=5.0):
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learn_rate)

        #if clip_norm:
        #    tvars = tf.trainable_variables()
        #    self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)
        #    self.grads_and_vars = zip(self.grads, tvars)
        #else:
        #    self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
        
        #self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        
        # see https://github.com/soloice/mnist-bn/blob/master/mnist_bn.py for an easy example on training with slim and batchnorm
        self.train_op = slim.learning.create_train_op(self.cost, self.optimizer, global_step=self.global_step)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops:
            print('Will add update_ops dependency ...')
            updates = tf.group(*self.update_ops)
            cross_entropy = control_flow_ops.with_dependencies([updates], self.cost)

        #self.train_op = self.optimizer.minimize(self.cost)

        #if FLAGS.log_tensorboard:
        #    # Keep track of gradient values and sparsity (optional)
        #    grad_summaries = []
        #    for g, v in self.grads_and_vars:
        #        if g is not None:
        #            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #            grad_summaries.append(grad_hist_summary)
        #            grad_summaries.append(sparsity_summary)
        #    grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries

        if create_new_train_dir:
            timestamp = str(int(time.time()))
            self.out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", timestamp)) + '/' + 'tf10'
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

        if FLAGS.log_tensorboard:
            
            loss_summary = tf.summary.scalar('loss', self.cost)
            self.train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(self.out_dir, "summaries", "train")

        # Dev summaries
        #dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #self.dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        #self.saver = tf.train.Saver(tf.global_variables()) 

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
            
            input_slice = np.array(audio_data[random_pos_num:random_pos_num+input_size])
            output_slice = np.array(audio_data[random_pos_num+input_size-1:random_pos_num+input_size+output_size])

            output_slice_dis = discretize(encode_mulaw(output_slice))
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

        # None -> automatically sets the dimension to batch_size
        # window length 80 sample = 5ms at 16kHz
        self.input_x = tf.placeholder(tf.float32, [None, window_length], name="input_x")
        # The true discrete labels of the sequence following the inputs
        self.decoder_inputs = tf.placeholder(tf.int32, [None, output_length], name="decoder_inputs")
        # The last symbol of the input sequence (discretized)
        self.input_symbol = tf.placeholder(tf.int32, [None, 1])

        #not used in training, but useful for sampling 
        self.input_y = tf.placeholder(tf.int32, [None, output_length], name="input_y")
        self.input_state = tf.placeholder(tf.float32, [None, fc_size*decoder_layers])
        
        #
        # The teacher forcing vector is an int vector with either zeros or ones. If the value x(t) is one at decoding step t, 
        # then the argmax of t-1 predicted output is used, otherwise the true previous label is used.
        # 
        # See also:
        # Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, "Scheduled Sampling for Sequence Prediction with
        # Recurrent Neural Networks", 2015 , (https://arxiv.org/pdf/1506.03099.pdf)
        #
        
        self.teacher_forcing = tf.placeholder(tf.float32, [None, output_length])

        second_cnn_layer = False

        with tf.variable_scope("unsupmodel"):
            #input_reshaped = tf.reshape(self.input_x, [-1, 1, window_length, 1])
            input_reshaped = tf.reshape(self.input_x, [-1, window_length, 1])

            print('input_shape:', input_reshaped)

            self.pooled_outputs = []

            #currently we only support one filtersize (but we could extend)
            #for i, filter_size in enumerate(filter_sizes):
            filter_size = filter_sizes[0]
            i=0

            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # 2D conv
                # [filter_height, filter_width, in_channels, out_channels]
                
                # 1D conv:
                # [filter_width, in_channels, out_channels]

                print('Filter size is:', filter_size)
                
                #this would be the filter for a conv2d:
                #filter_shape = [1 , filter_size, 1, num_filters]
                
                filter_shape = [filter_size, 1, num_filters]
                print('filter_shape:',filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                #W = tf.get_variable("W",shape=filter_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
                
                # tf.nn.atrous_conv2d?

                # 1D conv without padding(padding=VALID)
                #conv = tf.nn.conv2d(input_reshaped,W,strides=[1, 1, 2, 1],padding="VALID",name="conv")

                conv = tf.nn.conv1d(input_reshaped, W, stride=2, padding="VALID",name="conv1")

                with tf.variable_scope('visualization_conv1d'):
                    # scale weights to [0 1], type is still float
                    kernel_0_to_1 = tensor_normalize_0_to_1(W) 

                    # to tf.image_summary format [batch_size, height, width, channels]
                    kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
                    kernel_transposed = tf.expand_dims(kernel_transposed, 0)

                    # this will display random 3 filters from the 64 in conv1
                    tf.summary.image('conv1d_filters', kernel_transposed) #, max_images=3)

                ## Apply nonlinearity
                b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="bias1")
                conv = tf.nn.tanh(tf.nn.bias_add(conv, b), name="activation1")

                pool_input_dim = int(conv.get_shape()[1])

                print('pool input dim:', pool_input_dim)
                print('conv1 shape:',conv.get_shape())
                # Temporal maxpool accross all filters, pool size 2
                #pooled = tf.nn.max_pool(conv,ksize=[1, 1, pool_input_dim / 8, 1], # max_pool over / 4 of inputsize filters
                #                        strides=[1, 1, pool_input_dim / 16 , 1], # hopped by / 8 of input size
                #                        padding='VALID',name="pool")

                # check if the 1d pooling operation is correct
                pooled = pool1d(conv, ksize=[1, 4 , 1], strides=[1, 4 , 1], padding='VALID',name="pool")
                print('pool1 shape:',pooled.get_shape())

                pool_output_dim = int(pooled.get_shape()[1])
                print('pool_output_dim shape:',pooled.get_shape())

                pooled = tf.reshape(pooled,[-1,pool_output_dim, num_filters, 1])

                print('pool1 reshaped shape:',pooled.get_shape())

                #input shape: batch, in_height, in_width, in_channels
                #filter shape: filter_height, filter_width, in_channels, out_channels
                #('pool1 shape:', TensorShape([Dimension(None), Dimension(1), Dimension(7), Dimension(80)]))

                second_cnn_layer = True

                if second_cnn_layer:
                    
                    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm if FLAGS.batch_normalization else None,
                                                normalizer_params={'is_training': is_training, 'decay': 0.95}):
                        conv = DenseBlock2D(pooled, 10, 2, num_connected=3) #tf.nn.conv2d(pooled, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                        pooled = DenseTransition2D(conv, 40, 'transition1') 
                        
                        conv = DenseBlock2D(pooled, 10, 3, num_connected=3)
                        #pooled = DenseTransition2D(conv, 40, 'transition2')
                        pooled = DenseFinal2D(conv, 'dense_end')

                    print('pool shape after dense blocks:', pooled.get_shape())


                #with tf.variable_scope('visualization_embedding'):
                #    pooled_normalized = tensor_normalize_0_to_1(pooled)
                #    # tf.image_summary format [batch_size, height, width, channels]
                #    # example pool shape after dense blocks: (?, 8, 16, 10)
                #    tf.summary.image('learned_embedding', pooled_normalized, max_outputs=10)  

                flattened_size = int(pooled.get_shape()[1]*pooled.get_shape()[2]*pooled.get_shape()[3])
                # Reshape conv2 output to fit fully connected layer input
                self.flattened_pooled = tf.reshape(pooled, [-1, flattened_size])
            
                with tf.variable_scope('visualization_embedding'):
                    flattened_pooled_normalized = tensor_normalize_0_to_1(self.flattened_pooled)
                    tf.summary.image('learned_embedding', tf.reshape(flattened_pooled_normalized,[-1,1,flattened_size,1]), max_outputs=10)

                print('flattened_pooled shape:',self.flattened_pooled.get_shape())

                self.fc1 = self.fully_connected(self.flattened_pooled, flattened_size, fc_size*decoder_layers, name='fc1', use_dropout=False) #is_training)
                print('fc1 shape:',self.fc1.get_shape())
                
                #self.fc2 = self.fully_connected(self.fc1, fc_size*decoder_layers, fc_size*decoder_layers, name='fc2', use_dropout=is_training)
                #print('fc2 shape:',self.fc2.get_shape())

                #single_cell = tf.nn.rnn_cell.GRUCell(fc_size)
                #single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=0.8, state_is_tuple=False)
                print('Decoder size: %d, layers %d' % (fc_size , decoder_layers))
                #single_cell = tf.nn.rnn_cell.GRUCell(fc_size)
                #single_cell = tf.contrib.rnn.LSTMCell(fc_size, state_is_tuple=False)
                single_cell = GRUCell_MulInt(fc_size, use_recurrent_dropout=False) # is_training, recurrent_dropout_factor = 0.9 if is_training else 1.0)
                #cell = tf.contrib.rnn.MultiRNNCell([single_cell] * decoder_layers, state_is_tuple=False)
                
                cell = cwrnn.CWRNNCell([single_cell] * decoder_layers, [1,4,8,32,128,256,512,1024][:decoder_layers], state_is_tuple=False)
                
                #single_cell = HighwayRNNCell_MulInt(fc_size, num_highway_layers=decoder_layers, 
                #                use_recurrent_dropout=is_training, recurrent_dropout_factor=0.9)
                #cell = single_cell

                state = self.fc1
                self.initial_state = state                

                embedding = tf.get_variable("embedding", [mu+1, emb_size], initializer=tf.random_uniform_initializer(-1,1))
                self.decoder_inputs_emb = tf.nn.embedding_lookup(embedding, self.decoder_inputs)
                #self.decoder_first_input_emb = self.decoder_inputs_emb[:,0,:]
                self.input_symbol_emb = tf.nn.embedding_lookup(embedding, self.input_symbol)

                #next_input_emb = self.decoder_first_input_emb
                
                #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                
                #needed with clockwork rnns
                softmax_w = tf.get_variable("softmax_w", shape=[fc_size*decoder_layers, mu], dtype=tf.float32)
                #with standard RNN
                #softmax_w = tf.get_variable("softmax_w", shape=[fc_size, mu], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", shape=[mu], dtype=tf.float32)
                    
                if is_training:
                    rnn_outputs = []
                    #for training: multiple RNN steps
                    #with tf.variable_scope("decoderRNN"):
                    #    next_input_emb = self.decoder_inputs_emb[:,0,:]
                    #    (cell_output, state) = cell(next_input_emb, state)
                    #    logits = tf.matmul(cell_output, softmax_w) + softmax_b
                    #    rnn_outputs.append(logits)
                    print('decoder type is:', FLAGS.decoder_type)
                    if FLAGS.decoder_type == 'rnn':
                        with tf.variable_scope("decoderRNN"):
                            for time_step in range(0,output_length):
                                if time_step > 0: tf.get_variable_scope().reuse_variables()
                                #print('decoder_inputs shape:',self.decoder_inputs_emb[:,time_step,:].get_shape())
                                #print('state shape:',state.get_shape())
                                next_input_emb = self.decoder_inputs_emb[:,time_step,:]
                                if time_step > 0:
                                    output_symbol = tf.argmax(tf.nn.softmax(logits),1)
                                    # analog to numpys [:,time_step,np.newaxis] => shape is batchsize, 1
                                    teacher_force = tf.expand_dims(self.teacher_forcing[:,time_step], 1)
                                    # Either take the true label of t-1 (teacher forcing), or the argmax of the softmax distribution at timestep t-1, depending on the value in self.teacher_forcing[time_step] 
                                    next_input_emb = tf.multiply(next_input_emb,teacher_force) + tf.multiply(tf.nn.embedding_lookup(embedding, output_symbol), (1.0 - teacher_force))
                                if FLAGS.decoder_state_add_initial:
                                    (cell_output, state) = cell(next_input_emb, state + self.initial_state)
                                else:
                                    (cell_output, state) = cell(next_input_emb, state)
                                logits = tf.matmul(cell_output, softmax_w) + softmax_b        
                                rnn_outputs.append(logits)
                                #print('logits shape:',logits.get_shape())
                                #output_symbols = tf.argmax(tf.nn.softmax(logits),1)
                                #print('output_symbols shape:', output_symbols.get_shape())
                                #next_input_emb = tf.nn.embedding_lookup(embedding, output_symbols)
                                #print('next_input shape:', next_input_emb.get_shape())
                        rnn_output = tf.reshape(tf.concat(axis=1, values=rnn_outputs), [-1, mu])
                        print('rnn_output shape:',rnn_output.get_shape())
                        self.out = tf.reshape(tf.argmax(rnn_output,1),[-1, output_length])
                        print('out shape (argmaxes):',self.out.get_shape())

                        ce_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_output, labels=tf.reshape(self.input_y,[-1]))) / batch_size
                    elif FLAGS.decoder_type == 'nn':
                        with tf.variable_scope("nnDecoder"):
                            if output_length != 1:
                                print('Output size: ', output_length)
                                print('The nn decoder only makes sense with an output size of 1. Choose the rnn decoder or set output size to 1. Aborting.')
                                sys.exit()
                        output = tf.matmul(self.initial_state, softmax_w) + softmax_b
                        self.out = tf.reshape([tf.argmax(output,1)], [-1, output_length])
                        ce_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(output, [-1, mu]), labels=tf.reshape(self.input_y,[-1])))
                            

                    #[self.cell_output_softmax, self.output_state]
                   
                    ##single RNN step

                    if FLAGS.l2_regularization_strength is None:
                        self.cost = ce_cost
                    else:
                        print('imposing L2 reg on these vars:', ' '.join([v.name for v in tf.trainable_variables() if not('bias' in v.name)]))
                        # L2 regularization for all trainable parameters
                        l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                        # Add the regularization term to the loss
                        total_loss = (ce_cost +
                                  FLAGS.l2_regularization_strength * l2_loss)

                        tf.summary.scalar('l2_loss', l2_loss)
                        tf.summary.scalar('total_loss', total_loss)

                        self.cost = total_loss
                
                else:
                    with tf.variable_scope("decoderRNN"):
                        #                   with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                            #tf.get_variable_scope().reuse_variables()
                        (rnn_step_cell_output, rnn_step_state) = cell(self.input_symbol_emb[:,0,:], self.input_state)
                        self.cell_output = rnn_step_cell_output
                        self.output_state = rnn_step_state
                        self.cell_output_logits = tf.matmul(self.initial_state, softmax_w) + softmax_b
                        self.cell_output_softmax = tf.nn.softmax(self.cell_output_logits)
                
                if is_training:
                    self.create_training_graphs(create_new_train_dir)

                self.saver = tf.train.Saver(tf.global_variables())

    def run_rnn_step(self, sess, input_symbol, input_state, batch_size=1):
        data = np.zeros([batch_size, 1], dtype=np.int32)
        for i,symbol in enumerate(input_symbol):
            data[i] = symbol

        return sess.run([self.cell_output_softmax, self.output_state],{self.input_state: input_state,
                                      self.input_symbol: data})

    # do a training step with the supplied input data
    def step(self, sess, input_x, input_y, decoder_inputs, batch_size, current_step, lambd=0.00004):
        teacher_force = (np.random.rand(batch_size , self.output_length) < np.exp(-1.0 * lambd * current_step)) * 1.0
        feed_dict = {self.input_x: input_x, self.input_y: input_y, self.decoder_inputs: decoder_inputs, self.teacher_forcing: teacher_force}
        _, output, loss = sess.run([self.train_op, self.out, self.cost], feed_dict=feed_dict)
        return  output, loss, teacher_force

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

    def generate_signal(self, sess, np_signal, temperature=1.0, mulaw_signal=False):
        state = [self.gen_feat(sess, np_signal)]
        print("np_signal[-1]: ", np_signal[-1])
        print(state)
        
        #only discretize if signal is already in mulaw
        if mulaw_signal:
            input_symbol = discretize([np_signal[-1]])[0]
        else:
            input_symbol = discretize(encode_mulaw([np_signal[-1]]))[0]
            
        print("input_symbol: %d" % input_symbol)
        generated = []
        for i in xrange(self.output_length):
            cell_output, state = self.run_rnn_step(sess, [input_symbol], state)
            #input_symbol = np.argmax(cell_output[0])
            input_symbol = sample(cell_output[0],temperature=temperature)
            generated.append(input_symbol)
        print("Generated:",generated)
        return decode_mulaw(undiscretize(generated))

def gen_feat(filelist, sample_data=True, generate_challenge_output_feats=True, startpos_sample=20*16000-800):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/gpu:1'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_length=FLAGS.window_length, output_length=FLAGS.output_length, filter_sizes=filter_sizes,
                                num_filters=FLAGS.num_filters, fc_size=FLAGS.fc_size, dropout_keep_prob=1.0, train_files = filelist, create_new_train_dir = False, is_training=False, decoder_layers=FLAGS.decoder_layers)
            if FLAGS.train_dir != "":
                print('FLAGS.train_dir',FLAGS.train_dir)
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                print('ckpt:',ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    # model is now loaded with the trained parameters
                    for myfile in filelist:

                        if generate_challenge_output_feats:
                            input_signal = training_data[myfile]
                            hop_size = int(float(FLAGS.window_length) / 2.5)
                            print('Generate features for', myfile , 'window size:', FLAGS.window_length , 'hop size:', hop_size)
                            feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, int(float(FLAGS.window_length) / 2.5)))
                            utils.writeZeroSpeechFeatFile(feat, myfile.replace('.wav', '') + '.fea')

                        #testing to sample with the data at startpos_samples as warm start
                        if sample_data:
                            input_signal = training_data[myfile][startpos_sample:]
                            
                            if FLAGS.show_feat:
                                feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, 180)[:500])
                                pyplot.imshow(feat.T)
                                pyplot.show()
                                print(feat)
                                
                            pre_sig_length = 2000 #1450
                            gen_signal = input_signal[:pre_sig_length]
                            print('Generating signal...')
                            for i in xrange(FLAGS.gen_steps):
                                next_signal = model.generate_signal(sess, gen_signal[-FLAGS.window_length:], temperature=FLAGS.temp)
                                #model.gen_next_batch(sess, [gen_signal[-FLAGS.window_length:]])
                                #input_signal = input_signal[FLAGS.output_length:] + next_signal[0]
                                if i % 100 == 0:
                                    print(next_signal[0])
                                    print(gen_signal.shape)
                                gen_signal = np.append(gen_signal,next_signal)
                            utils.writeSignal(gen_signal, FLAGS.genwav_dir + '/gentest_o'+str(FLAGS.output_length)+'.temp'+str(FLAGS.temp)+'.gen_steps'+str(FLAGS.gen_steps)+'.wav')
                            print('done!')
                            break
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
                    print("Couldn't load parameters from:" + FLAGS.train_dir)
            if not restored:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            summary_writer = None
            if FLAGS.log_tensorboard:
                summary_writer = tf.summary.FileWriter(model.out_dir, sess.graph)

            #write out configuration
            with open(model.out_dir + '/tf_param_train', 'w') as tf_param_train:
                tf_param_train.write(get_FLAGS_params_as_str())

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
                    summary_str = sess.run(model.train_summary_op, feed_dict={model.input_x:input_x, model.input_y:input_y, model.decoder_inputs: decoder_inputs, model.teacher_forcing: teacher_force})
                    summary_writer.add_summary(summary_str, current_step)

                # Get a batch and make a step.
                start_time = time.time()
                input_x, input_y, decoder_inputs = model.get_batch(filelist=filelist, input_size=FLAGS.window_length, output_size=FLAGS.output_length, batch_size=FLAGS.batch_size, model = 'post_pred')
                output_y, train_loss, teacher_force = model.step(sess, input_x, input_y, decoder_inputs, FLAGS.batch_size , current_step)
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
                    print('teacher_force:', teacher_force[0])
                    print('At step %i step-time %.4f loss %.4f' % (current_step, step_time, mean_train_loss))
                    #print('Input_x, input_y:', input_x[0], input_y[0])
                    train_losses = []
                    energy_losses = []
                    step_time = 0
                    if checkpoint_step % FLAGS.checkpoints_per_save == 0:
                        min_loss = 1e10
                        if len(previous_losses) > 0:
                            min_loss = min(previous_losses)
                        #if 1==1:
                        if mean_train_loss < min_loss:
                            print(('Train loss: %.6f' % mean_train_loss) + (' is smaller than previous best loss: %.6f' % min_loss) )
                            print('Saving the best model so far to ', model.out_dir, '...')
                            model.saver.save(sess, model.out_dir, global_step=model.global_step)
                            previous_losses.append(mean_train_loss)


if __name__ == "__main__":
    FLAGS._parse_flags()
    print("\nParameters:")
    print(get_FLAGS_params_as_str())
    filelist = utils.loadIdFile(FLAGS.filelist, 300)
    print(filelist)

    x = np.linspace(-4, 4, 41)
    x = np.sin(x)

    print('signal:',x)
    y = encode_mulaw(x)
    print('encode_mulaw:',y)
    x = decode_mulaw(y)
    print('decode_mulaw:', x)
    dis = discretize(x)
    undis = undiscretize(dis)

    print('discretize:',dis)
    print('undiscretize:',undis)

    print('continuing training in 5 seconds...')
    time.sleep(5)

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
        #signal = encode_mulaw(signal)
        #signal /= np.std(signal)
        #signal = (signal-np.mean(signal))/np.std(signal);
        
        training_data[myfile] = signal

    if FLAGS.eval:
        print(gen_feat([filelist[-1]]))
    else:
        train(filelist)
