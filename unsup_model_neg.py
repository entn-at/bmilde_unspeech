#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:29:11 2017

@author: me
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import sys
import time
import utils
import math
import kaldi_io
import itertools
import pylab as plt

from sklearn.metrics import accuracy_score

#from tensorflow.python.ops import control_flow_ops

if sys.version_info[0] == 3:
    xrange = range

tf.flags.DEFINE_string("filelist", "filelist.english.train", "Kaldi scp file if using kaldi feats, or for end-to-end learning a simple filelist, one wav file per line, optionally also an id (id wavfile per line).")
tf.flags.DEFINE_string("spk2utt", "", "Optional, but required for per speaker negative sampling. ")
tf.flags.DEFINE_boolean("end_to_end", False, "Use end-to-end learning (Input is 1D). Otherwise input is 2D like FBANK or MFCC features.")
tf.flags.DEFINE_boolean("feat_size", 40, "Size of the features inner dimension (only used if not using end-to-end training).")

tf.flags.DEFINE_boolean("debug", False, "Limits the filelist size and is more debug.")

tf.flags.DEFINE_boolean("gen_feats", False, "Load a model from train_dir")
tf.flags.DEFINE_boolean("test_sampling", False, "Test the sampling algorithm")

tf.flags.DEFINE_boolean("generate_kaldi_output_feats", False, "Whether to write out a feature file for Kaldi (containing all utterances), requires a trained model")
tf.flags.DEFINE_string("output_kaldi_ark", "output_kaldi.ark" , "Output file for Kaldi ark file")
tf.flags.DEFINE_boolean("generate_challenge_output_feats", False, "Whether to write out a feature file in the unsupervise vhallenge format (containing all utterances), requires a trained model")
tf.flags.DEFINE_integer("hop_size", 200,"How many training steps to do per checkpoint.")
tf.flags.DEFINE_string("model_name", "feat1", "Model output name, currently only used for generate_challenge_output_feats")

tf.flags.DEFINE_integer("sample_rate", 16000, "Sample rate of the audio files. Must have the same samplerate for all audio files.") # 100+ ms @ 16kHz
tf.flags.DEFINE_string("filter_sizes", "512", "Comma-separated filter sizes (default: '200')") # 25ms @ 16kHz
tf.flags.DEFINE_integer("num_filters", 40, "Number of filters per filter size (default: 40)")

tf.flags.DEFINE_integer("window_length", 50, "Main window length, samples (end-to-end) or frames (FBANK)") # 100+ ms @ 16kHz
tf.flags.DEFINE_integer("window_neg_length", 50, "Context window length, samples (end-to-end) or frames (FBANK)") # 100+ ms @ 16kHz

tf.flags.DEFINE_integer("left_contexts", 2, "How many left context windows")
tf.flags.DEFINE_integer("right_contexts", 2, "How many right context windows")

tf.flags.DEFINE_integer("embedding_size", 100 , "Fully connected size at the end of the network.")

tf.flags.DEFINE_integer("fc_size", 512 , "Fully connected size at the end of the network.")

tf.flags.DEFINE_boolean("first_layer_tanh", True, "Whether tanh should be used for the output conv1d filters in end-to-end networks.")
tf.flags.DEFINE_boolean("first_layer_log1p", True, "Whether log1p should be applied to the output of the conv1d filters.")

tf.flags.DEFINE_string("embedding_transformation", "BaselineDnn", "What network to use for the embeddings computation. Vgg16, DenseNet, BaselineDnn, HighwayDnn.")

tf.flags.DEFINE_integer("dense_block_filters", 5,  "Number of filters inside a conv2d in a dense block.")
tf.flags.DEFINE_integer("dense_block_layers_connected", 3,  "Number of layers inside dense block.")
tf.flags.DEFINE_integer("dense_block_filters_transition", 4, "Number of filters inside a conv2d in a dense block transition.")

tf.flags.DEFINE_integer("num_highway_layers", 6, "How many layers for the highway dnn.")
tf.flags.DEFINE_integer("num_dnn_layers", 3, "How many layers for the baseline dnn.")

tf.flags.DEFINE_boolean("tied_embeddings_transforms", False, "Whether the transformations of the embeddings windows should have tied weights. Only makes sense if the window sizes match.")
tf.flags.DEFINE_boolean("use_weighted_loss_func", False, "Whether the class imbalance of having k negative samples should be countered by weighting the positive examples k-times more.")
tf.flags.DEFINE_boolean("use_dot_combine", True, "Define the loss function over the logits of the dot product of window and context window.")

tf.flags.DEFINE_integer("negative_samples", 4, "How many negative samples to generate.")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("batch_normalization", False, "Whether to use batch normalization.")

tf.flags.DEFINE_float("dropout_keep_prob", 1.0 , "Dropout keep probability")

tf.flags.DEFINE_integer("steps_per_checkpoint", 400,
                                "How many training steps to do per checkpoint.")
tf.flags.DEFINE_integer("steps_per_summary", 200,
                                "How many training steps to do per checkpoint.")

tf.flags.DEFINE_integer("checkpoints_per_save", 1,
                                "How many checkpoints until saving the model.")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_float("learn_rate", 1e-4, "Learn rate for the optimizer")
tf.flags.DEFINE_float("gradient_clipping", 5.0, "Clip the gradient at larger +/- this value.")

tf.flags.DEFINE_boolean("log_tensorboard", True, "Log training process if this is set to True.")

tf.flags.DEFINE_string("train_dir", "/srv/data/milde/unspeech_models/neg/", "Training dir to resume training from. If empty, a new one will be created.")
tf.flags.DEFINE_string("output_feat_file", "/srv/data/milde/unspeech_models/feats/", "Necessary suffixes will get appended (depending on output format).")
tf.flags.DEFINE_string("output_feat_format", "kaldi_bin", "Feat format")
tf.flags.DEFINE_string("genfeat_hopsize", 160, "Hop size (in samples if end-to-end) for the feature generation.")

FLAGS = tf.flags.FLAGS

training_data = {}
#TODO load this correctly. Format: dict, spk -> list utt ids
spk2utt_data = {}

def get_FLAGS_params_as_str():
    params_str = ''
    for attr, value in sorted(FLAGS.__flags.items()):
        params_str += "{}={}\n".format(attr.upper(), value)
    return params_str

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

#https://gist.github.com/awjuliani/fb10d1ea206fab25f946512d959e3894
def DenseBlock2D(input_layer,filters, layer_num, num_connected, non_linearity=lrelu):
    name = "dense_unit"+str(layer_num)
    with tf.variable_scope(name):
        nodes = []
        a = slim.conv2d(input_layer,filters,[3,3], activation_fn=non_linearity, scope=name+'0')
        nodes.append(a)
        for z in range(num_connected):
            b = slim.conv2d(tf.concat(nodes,3),filters,[3,3], activation_fn=non_linearity, scope=name+str(z+1))
            nodes.append(b)
        return b

#https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py
def DenseTransition2D(l, filters, name, with_conv=True, non_linearity=lrelu):
    with tf.variable_scope(name):
        if with_conv:
            l = slim.conv2d(l,filters,[3,3], activation_fn=non_linearity)
        l = slim.avg_pool2d(l, [2,2])
    return l

def DenseFinal2D(l, name, pool_size=7):
    with tf.variable_scope(name):
        l = slim.avg_pool2d(l, [pool_size,pool_size], stride=1)
    return l

#from https://github.com/tensorflow/tensorflow/tree/r1.2/tensorflow/contrib/slim
def vgg16(inputs):
#  with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                      activation_fn=lrelu,
#                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#                      weights_regularizer=slim.l2_regularizer(0.0005),
#                      biases_initializer = tf.constant_initializer(0.01)):
    print('vgg input shape:',inputs.get_shape())
    net = slim.repeat(inputs, 2, slim.conv2d, 16, [3, 3], scope='conv1')
    print('vgg input conv1 shape:', net.get_shape())
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    print('vgg input pool1 shape:', net.get_shape())
    net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2')
    print('vgg input conv2 shape:', net.get_shape())
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    print('vgg input pool2 shape:', net.get_shape())
    net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv3')
    print('vgg input conv3 shape:', net.get_shape())
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    print('vgg input pool3 shape:', net.get_shape())
    net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv4')
    print('vgg input conv4 shape:', net.get_shape())
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    print('vgg input pool4 shape:', net.get_shape())
    #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    #net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #net = slim.fully_connected(net, 512, scope='fc6')
    #print('vgg input fc1 shape:', net.get_shape())
    #net = slim.dropout(net, 0.5, scope='dropout6')
    #net = slim.fully_connected(net, 256, scope='fc7')
    #print('vgg input fc2 shape:', net.get_shape())
    #net = slim.dropout(net, 0.5, scope='dropout7')
    #net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net

# highway impl from https://github.com/fomorians/highway-fcn/blob/master/main.py

def weight_bias(W_shape, b_shape, bias_init=0.1, stddev=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def highway_layer(x, size, activation, carry_bias=-1.0):
    W, b = weight_bias([size, size], [size])

    with tf.name_scope('transform_gate'):
        W_T, b_T = weight_bias([size, size],[size], bias_init=carry_bias)

    H = activation(tf.matmul(x, W) + b, name='activation')
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    C = tf.subtract(1.0, T, name="carry_gate")

    y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y') # y = (H * T) + (x * C)
    return y


def get_random_audiosample(idlist, idlist_size, window_size, random_id=None, spk_id=None, spk2utt=None, spk2utt_keys= None , num_speakers = 0, spk2len=None, debug=False):

    # sample over a random file, if no specific one was specified
    if spk_id is None and random_id is None:
        random_id_num = int(math.floor(np.random.random_sample() * float(idlist_size)))
        # if spk2utt is supplied, we sample a speaker first, then a utterance id
        if spk2utt is not None:
            random_spk_num = int(math.floor(np.random.random_sample() * float(num_speakers)))
            spk_id = spk2utt_keys[random_spk_num]
            random_id_num = int(math.floor(np.random.random_sample() * float(spk2len[spk_id])))
            random_id = spk2utt[spk_id][random_id_num]
            if debug:
                print('[get_random_audiosample] Selecting sample:', 'speaker:',spk_id, 'uttid:', random_id)

        else:
            random_id = idlist[random_id_num]
    else:
        if spk_id is not None:
            print(spk2utt[spk_id],spk2len[spk_id])
            random_id_num = int(math.floor(np.random.random_sample() * float(spk2len[spk_id])))
            random_id = spk2utt[spk_id][random_id_num]  
        else:
            random_id = idlist[random_id_num]
        if debug:
            print('[get_random_audiosample] Selecting sample:', 'speaker:',spk_id, 'uttid:', random_id)

        
    audio_data = training_data[random_id]
    audio_len = audio_data.shape[0] - window_size
    random_pos_num = int(math.floor(np.random.random_sample() * audio_len))
    
    if debug:
        print('[get_random_audiosample] Select position:', random_pos_num,'window_size:',window_size)
    
    return np.array(audio_data[random_pos_num:random_pos_num+window_size]), spk_id
   

# does a batch where one of the examples are two windows with consecutive signals and k randomly selected window_2s
#, with a fixed window1
def get_batch_k_samples(idlist, window_length, window_neg_length, spk2utt=None, spk2len=None, num_speakers = 0, left_contexts=0, right_contexts=1 , k=4, debug=False):            
    window_batch = []
    window_neg_batch = []
    labels = []
    
    idlist_size = len(idlist)
    spk_id=None
    spk2utt_keys = list(spk2utt.keys())
    
    for i in xrange(FLAGS.batch_size*(k+1)):
        if i%(k+1)==0:
            if debug:
                print('Selecting true sample.')
            # Get a large sample to fit all the windows. Also get a speaker id, if we have speaker information (spk2utt!=None) and want to sample per speaker. 
            combined_sample, spk_id = get_random_audiosample(idlist, idlist_size, window_length+window_neg_length*(left_contexts+right_contexts), spk2utt=spk2utt, spk2utt_keys=spk2utt_keys,  num_speakers=num_speakers, spk2len=spk2len)
            # Getting all the context pairs, e.g. context_num goes from -2 to 2 for left_contexts=2 and right_contexts=2
            center_window_pos = window_neg_length*left_contexts
            for context_num in xrange(-1*left_contexts, right_contexts+1):
                if context_num < 0:
                    neg_pos = (left_contexts+context_num)*window_neg_length
                elif context_num > 0:
                    neg_pos = center_window_pos+window_length+(context_num-1)*window_neg_length
                if context_num !=0:
                    window = combined_sample[center_window_pos:center_window_pos+window_length]
                    window_neg = combined_sample[neg_pos:neg_pos+window_neg_length] 
                    # Assign label 1, if both windows are consecutive    
                    labels.append(1.0)
                    window_batch.append(window)
                    window_neg_batch.append(window_neg)
        else:
            
            if debug:
                print('Selecting random sample from speaker:',spk_id)
            # Select two random samples. If we do per speaker sampling, then spk_id!= None and two samples from the same speaker are sampled.
            # Otherwise, if random_file_num is not None, we do per utterance sampling.
            window, _ = get_random_audiosample(idlist, idlist_size, window_length, random_id=None, spk2utt=spk2utt, spk_id=spk_id, spk2utt_keys=spk2utt_keys, num_speakers=num_speakers, spk2len=spk2len)
            window_neg, _ = get_random_audiosample(idlist, idlist_size, window_neg_length, random_id=None, spk2utt=spk2utt, spk_id=spk_id, spk2utt_keys=spk2utt_keys, num_speakers=num_speakers, spk2len= spk2len)
            # Assign label 0, if both windows are randomly sampled
            labels.append(0.0)
            
            window_batch.append(window)
            window_neg_batch.append(window_neg)

    labels = np.asarray(labels).reshape(-1,1)

    #if self.first_call_to_get_batch:
    #    print("window_batch,",[elem[:5] for elem in window_batch],"window_neg_batch,",[elem[:5] for elem in window_neg_batch],"labels",labels) 
    #    self.first_call_to_get_batch = False

    return window_batch,window_neg_batch,labels

class UnsupSeech(object):
    """
    Unsupervised learning with RAW speech signals. This model learns a speech representation by u
    using a negative sampling objective, where true contexts must be discrimnated from samples ones
    """
    
    def create_training_graphs(self, create_new_train_dir=True, clip_norm=True, max_grad_norm=5.0):
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learn_rate)                
        
        #self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#        if self.update_ops:
 #           print('Will add update_ops dependency ...')
  #          updates = tf.group(*self.update_ops)
   #         self.opt_cost = control_flow_ops.with_dependencies([updates], self.cost)
    #    else:
     #       self.opt_cost = self.cost
            
        self.train_op = slim.learning.create_train_op(self.cost, self.optimizer, global_step=self.global_step, clip_gradient_norm=FLAGS.gradient_clipping)

        if create_new_train_dir:
            timestamp = str(int(time.time()))
            self.out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", timestamp)) + '/' + 'tf10'
            print("Writing to {}\n".format(self.out_dir))
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            #checkpoint_prefix = os.path.join(checkpoint_dir, "model")
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
    
    def __init__(self, window_length, window_neg_length, filter_sizes, num_filters, fc_size, embeddings_size, dropout_keep_prob, train_files, k, left_contexts, right_contexts, is_training=True, create_new_train_dir=True, batch_size=128, feat_size=40):

        self.train_files = train_files

        self.window_length = window_length
        self.window_neg_length = window_neg_length
        self.fc_size = fc_size
        self.embeddings_size = embeddings_size
        self.left_contexts = left_contexts
        self.right_contexts = right_contexts

        # feat_size of 0 means were gonig end-to-end with 1d samples
        if feat_size == 0:
            # window 1 is fixed
            self.input_window_1 = tf.placeholder(tf.float32, [None, window_length], name="input_window_1")
            # window 2 is either consecutive, or randomly sampled
            self.input_window_2 = tf.placeholder(tf.float32, [None, window_neg_length], name="input_window_2")
        else:
            # standard speech features, e.g. FBANK in 2D
            # window 1 is fixed
            self.input_window_1 = tf.placeholder(tf.float32, [None, window_length, feat_size], name="input_window_1")
            # window 2 is either consecutive, or randomly sampled
            self.input_window_2 = tf.placeholder(tf.float32, [None, window_neg_length, feat_size], name="input_window_2")
        
        self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        
        self.first_call_to_get_batch = True
        
        with slim.arg_scope([slim.conv2d, slim.fully_connected],  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            #weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            weights_regularizer=slim.l2_regularizer(0.0005),
                                            activation_fn=lrelu,
                                            biases_initializer = tf.constant_initializer(0.01),
                                            normalizer_fn=slim.batch_norm if FLAGS.batch_normalization else None,
                                            normalizer_params={'is_training': is_training, 'decay': 0.95} if FLAGS.batch_normalization else None):
            with tf.variable_scope("unsupmodel"):
                # a list of embeddings to use for the binary classifier (the embeddings are combined)
                self.outs = []
                for i,input_window in enumerate([self.input_window_1, self.input_window_2]):
                    with tf.variable_scope("embedding-transform" if FLAGS.tied_embeddings_transforms else "embedding-transform-" + str(i)):     
                        if FLAGS.tied_embeddings_transforms and i > 0: 
                            print("Reusing variables for embeddings computation.")
                            tf.get_variable_scope().reuse_variables()
                        #input_reshaped = tf.reshape(self.input_x, [-1, 1, window_length, 1])
                        window_length = int(input_window.get_shape()[1])
                        
                        if feat_size == 0:
                            #1conv over 1d samples
                            input_reshaped = tf.reshape(input_window, [-1, window_length, 1])
                
                            print('input_shape:', input_reshaped)
                
                            self.pooled_outputs = []
                
                            #currently we only support one filtersize (but we could extend)
                            #for i, filter_size in enumerate(filter_sizes):
                            filter_size = filter_sizes[0]
        
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
                            
            
                            # 1D conv without padding(padding=VALID)
                            #conv = tf.nn.conv2d(input_reshaped,W,strides=[1, 1, 2, 1],padding="VALID",name="conv")
            
                            conv = tf.nn.conv1d(input_reshaped, W, stride=2, padding="VALID",name="conv1")
            
                            with tf.variable_scope('visualization_conv1d'):
                                # scale weights to [0 1], type is still float
                                kernel_0_to_1 = utils.tensor_normalize_0_to_1(W) 
            
                                # to tf.image_summary format [batch_size, height, width, channels]
                                kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
                                kernel_transposed = tf.expand_dims(kernel_transposed, 0)
            
                                # this will display random 3 filters from the 64 in conv1
                                tf.summary.image('conv1d_filters', kernel_transposed) #, max_images=3)
            
                            ## Apply nonlinearity
                            b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="bias1")
                            
                            if FLAGS.first_layer_tanh:
                                conv = tf.nn.tanh(tf.nn.bias_add(conv, b), name="activation1")
                            else:
                                conv = lrelu(tf.nn.bias_add(conv, b), name="activation1")
            
                            if FLAGS.first_layer_log1p:
                                conv = tf.log1p(tf.abs(conv))
            
                            pool_input_dim = int(conv.get_shape()[1])
            
                            print('pool input dim:', pool_input_dim)
                            print('conv1 shape:',conv.get_shape())
                            # Temporal maxpool accross all filters, pool size 2
                            #pooled = tf.nn.max_pool(conv,ksize=[1, 1, pool_input_dim / 8, 1], # max_pool over / 4 of inputsize filters
                            #                        strides=[1, 1, pool_input_dim / 16 , 1], # hopped by / 8 of input size
                            #                        padding='VALID',name="pool")
            
                            # check if the 1d pooling operation is correct
                            pooled = pool1d(conv, ksize=[1, 2 , 1], strides=[1, 2 , 1], padding='VALID',name="pool")
                            print('pool1 shape:',pooled.get_shape())
            
                            pool_output_dim = int(pooled.get_shape()[1])
                            print('pool_output_dim shape:',pooled.get_shape())
            
                            pooled = tf.reshape(pooled,[-1,pool_output_dim, num_filters, 1])
                        else:
                            pooled = tf.reshape(input_window, [-1, window_length , FLAGS.feat_size ,1])
                            
                        print('net input shape:',pooled.get_shape())
        
                        #input shape: batch, in_height, in_width, in_channels
                        #filter shape: filter_height, filter_width, in_channels, out_channels
                        #('pool1 shape:', TensorShape([Dimension(None), Dimension(1), Dimension(7), Dimension(80)]))
            
                        needs_flattening = True
                        if FLAGS.embedding_transformation == "DenseNet":
                            
                            #with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            #                    weights_regularizer=slim.l2_regularizer(0.0005),
                            #                    biases_initializer = tf.constant_initializer(0.01) if not FLAGS.batch_normalization else None,
                            #                    normalizer_fn=slim.batch_norm if FLAGS.batch_normalization else None,
                            #                    normalizer_params={'is_training': is_training, 'decay': 0.95} if FLAGS.batch_normalization else None):
                                
                                #input_layer,filters, layer_num, num_connected, non_linearity=lrelu
                            conv = DenseBlock2D(input_layer=pooled, filters=FLAGS.dense_block_filters, layer_num=2, num_connected=FLAGS.dense_block_layers_connected) #tf.nn.conv2d(pooled, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                            pooled = DenseTransition2D(l=conv, filters=FLAGS.dense_block_filters_transition, name='transition1', with_conv=True) 
                            
                            conv = DenseBlock2D(pooled, filters=FLAGS.dense_block_filters, layer_num=3, num_connected=FLAGS.dense_block_layers_connected)
                            #pooled = DenseTransition2D(conv, 40, 'transition2')
                            pooled = DenseFinal2D(conv, name='dense_end')
        
                            print('pool shape after dense blocks:', pooled.get_shape())
        
                        if FLAGS.embedding_transformation == "Vgg16":
                            pooled = vgg16(pooled)
                            print('pool shape after vgg16 block:', pooled.get_shape())
        
                        if needs_flattening:
                            flattened_size = int(pooled.get_shape()[1]*pooled.get_shape()[2]*pooled.get_shape()[3])
                            # Reshape conv2 output to fit fully connected layer input
                            self.flattened_pooled = tf.reshape(pooled, [-1, flattened_size])
                        else:
                            self.flattened_pooled = pooled
                        
                        if FLAGS.embedding_transformation == "HighwayDnn":
                            self.flattened_pooled = slim.fully_connected(self.flattened_pooled, fc_size*4)
                            for x in range(FLAGS.num_highway_layers):
                                self.flattened_pooled = highway_layer(self.flattened_pooled, fc_size*4, lrelu, carry_bias=-1.0)
                            
                        if FLAGS.embedding_transformation == "BaselineDnn":
                            #with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                                                #weights_regularizer=slim.l2_regularizer(0.0005),
                                                #biases_initializer = tf.constant_initializer(0.01) if not FLAGS.batch_normalization else None,
                                                #normalizer_fn=slim.batch_norm if FLAGS.batch_normalization else None,
                                                #normalizer_params={'is_training': is_training, 'decay': 0.95} if FLAGS.batch_normalization else None):
                            for x in range(FLAGS.num_dnn_layers):
                                self.flattened_pooled = slim.fully_connected(self.flattened_pooled, self.fc_size)                    
                   
                        #with tf.variable_scope('visualization_embedding'):
                        #    flattened_pooled_normalized = utils.tensor_normalize_0_to_1(self.flattened_pooled)
                        #    tf.summary.image('learned_embedding', tf.reshape(flattened_pooled_normalized,[-1,1,flattened_size,1]), max_outputs=10)
        
                        print('flattened_pooled shape:',self.flattened_pooled.get_shape())
        
                        self.fc1 = slim.fully_connected(self.flattened_pooled, self.fc_size)#weights_initializer=tf.truncated_normal_initializer(stddev=0.01)) #is_training)
                        print('fc1 shape:',self.fc1.get_shape())
                        self.outs.append(self.fc1)
                
                if FLAGS.use_dot_combine:
                    # computes the dot product between self.outs[0] and self.outs[1]
                    self.logits = tf.reduce_sum( tf.multiply(self.outs[0], self.outs[1]), 1, keep_dims=True)
                else:
                    # this is an alternative formulation that substracts self.outs[0] and self.outs[1] and projects down to 1
                    stacked = self.outs[0] - self.outs[1] #tf.concat(self.outs, 1)
                    print('stacked shape:',stacked.get_shape())
                    
                    self.logits = slim.fully_connected(stacked, self.embeddings_size)
                    self.logits = slim.fully_connected(self.logits, 1, activation_fn=None)#weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                
                if FLAGS.use_weighted_loss_func and (k != self.left_contexts + self.right_contexts):
                    # the goal of the weighting is do counterbalance class imbalances, so that negative and positive examples have a 50% weight in the final loss each
                    # the weighting computation only works, if there is a counter balance:
                    neg_coef = float(k) / float(self.left_contexts + self.right_contexts)
                    self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels, logits=self.logits, pos_weight=(neg_coef-1.0)*self.labels+1.0))
                else:
                    self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        
                self.out = tf.nn.sigmoid(self.logits)
        
                if is_training:
                    self.create_training_graphs(create_new_train_dir)
                
                self.saver = tf.train.Saver(tf.global_variables())

    # do a training step with the supplied input data
    def step(self, sess, input_window_1, input_window_2, labels):
        feed_dict = {self.input_window_1: input_window_1, self.input_window_2: input_window_2, self.labels: labels}
        _, output, loss = sess.run([self.train_op, self.out, self.cost], feed_dict=feed_dict)
        return  output, loss

    def gen_feat_batch(self, sess, input_windows):
        feed_dict = {self.input_window_1: input_windows}
        feats = sess.run(self.outs[0], feed_dict=feed_dict)
        return feats

def get_model_flags_param_short():
    ''' get model params as string, e.g. to use it in an output filepath '''
    return ('e2e' if FLAGS.end_to_end else '') + '_trans' + FLAGS.embedding_transformation + '_win' + str(FLAGS.window_length) + '_lcontexts' + str(FLAGS.left_contexts) + '_rcontexts' + str(FLAGS.right_contexts) + \
                                    '_flts' + str(FLAGS.num_filters) + '_embsize' + str(FLAGS.embedding_size) + ('_dnn' + str(FLAGS.num_dnn_layers) if FLAGS.embedding_transformation=='BaselineDnn' else '') + \
                                    ('_highwaydnn' + str(FLAGS.num_highway_layers) if FLAGS.embedding_transformation=='HighwayDnn' else '') + \
                                    ('_dot_combine' if FLAGS.use_dot_combine else '')
    
def gen_feat(utt_id_list, filelist, feats_outputfile, feats_format, hop_size):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/gpu:1'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_length=FLAGS.window_length, window_neg_length=FLAGS.window_neg_length, filter_sizes=filter_sizes, 
                    num_filters=FLAGS.num_filters, fc_size=FLAGS.fc_size, emeddings_size=FLAGS.emeddings_size, dropout_keep_prob=FLAGS.dropout_keep_prob, k = FLAGS.negative_samples, 
                    left_contexts=FLAGS.left_contexts, right_contexts=FLAGS.right_contexts, train_files = utt_id_list,  batch_size=FLAGS.batch_size, is_training=False, create_new_train_dir=False)
            
            if FLAGS.train_dir != "":
                print('FLAGS.train_dir',FLAGS.train_dir)
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                print('ckpt:',ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    first_file = True
                    
                    window_length_seconds = float(FLAGS.window_length)/float(FLAGS.sample_rate)
                    model_params = get_model_flags_param_short()
                    
                    outputfile = feats_outputfile.replace('%model_params', model_params)
                    
                    utils.ensure_dir(outputfile)
                    
                    # model is now loaded with the trained parameters
                    for myfile in filelist:

                        if feats_format == "unsup_challenge2017":
                            input_signal = training_data[myfile]
                            hop_size = int(float(FLAGS.window_length) / 2.5)
                            print('Generate features for', myfile , 'window size:', FLAGS.window_length , 'hop size:', hop_size)
                            hop_size_seconds = float(hop_size)/float(FLAGS.sample_rate)
                            feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, hop_size))
                            out_filename = myfile.replace('.wav', '').replace('zerospeech2017/','zerospeech2017/'+FLAGS.model_name+model_params+'/') + '.fea'
                            print('Writing to ', out_filename)
                            utils.writeZeroSpeechFeatFile(feat, out_filename, window_length_seconds, hop_size_seconds )

                        if feats_format == "kaldi_text":
                            input_signal = training_data[myfile]
                            hop_size = FLAGS.kaldi_hopsize
                            print('Generate KALDI text features for', myfile , 'window size:', FLAGS.window_length , 'hop size:', hop_size)
                            feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, hop_size))
                            utils.writeArkTextFeatFile(feat,  myfile.replace('.wav', '') , FLAGS.output_kaldi_ark, not first_file)
                            first_file = False
                            
                        if feats_format == "kaldi_bin":           
                            input_signal = training_data[myfile]
                            
                            print('Generate KALDI bin features for', myfile , 'window size:', FLAGS.window_length , 'hop size:', hop_size)
                            feat = model.gen_feat_batch(sess, utils.rolling_window(input_signal, FLAGS.window_length, hop_size))
                            print('Done, writing to ' + outputfile)
                            pointers = kaldi_io.writeArk(outputfile + '.ark', [feat], [file2id[myfile]], append = not first_file)
                            kaldi_io.writeScp(outputfile + '.scp', [file2id[myfile]], pointers, append=not first_file)
                            first_file = False

                else:
                    print("Could not open training dir: %s" % FLAGS.train_dir)
            else:
                print("Train_dir parameter is empty")    
    
def train(utt_id_list, spk2utt=None, spk2len=None, num_speakers=None):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/gpu:1'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_length=FLAGS.window_length, window_neg_length=FLAGS.window_neg_length, filter_sizes=filter_sizes, 
                                num_filters=FLAGS.num_filters, fc_size=FLAGS.fc_size, embeddings_size = FLAGS.embedding_size, dropout_keep_prob=FLAGS.dropout_keep_prob, 
                                k = FLAGS.negative_samples, left_contexts=FLAGS.left_contexts, right_contexts=FLAGS.right_contexts, train_files = utt_id_list,  batch_size=FLAGS.batch_size)
            
            training_start_time = time.time()
            restored = False
            if FLAGS.train_dir != "":
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
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

            step_time = 0.0
            current_step = 0
            checkpoint_step = 0
            previous_losses = []
            input_window_1, input_window_2, labels = None, None, None

            while True:
                current_step += 1

                if current_step % FLAGS.steps_per_summary == 0 and summary_writer is not None:
                    #input_window_1, input_window_2, labels = model.get_batch_k_samples(filelist=filelist, window_length=FLAGS.window_length, window_neg_length=FLAGS.window_neg_length, k=FLAGS.negative_samples)
                    summary_str = sess.run(model.train_summary_op, feed_dict={model.input_window_1:input_window_1,
                                                                              model.input_window_2:input_window_2, model.labels: labels})
                    summary_writer.add_summary(summary_str, current_step)

                # Get a batch and make a step.
                start_time = time.time()
                input_window_1, input_window_2, labels = get_batch_k_samples(idlist=utt_id_list, window_length=FLAGS.window_length, 
                                                                                   window_neg_length=FLAGS.window_neg_length, left_contexts=FLAGS.left_contexts,
                                                                                   right_contexts=FLAGS.right_contexts, k=FLAGS.negative_samples, 
                                                                                   spk2utt=spk2utt, spk2len=spk2len , num_speakers=num_speakers)
                
                out, train_loss = model.step(sess, input_window_1, input_window_2, labels)
                train_losses.append(train_loss)

                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    checkpoint_step += 1
                    mean_train_loss = np.mean(train_losses)

                    #print('input_window_1:', input_window_1[0])
                    #print('input_window_2:', input_window_2[0])
                    labels_len = len(labels)
                    out_len = len(out)
                    
                    labels_flat = np.reshape(labels,[-1])
                    out_flat = (np.reshape(out,[-1]) > 0.5) * 1.0
                    out_flat_zero = np.zeros_like(labels_flat)
                    
                    print('np.bincount:', np.bincount(out_flat.astype('int32')))
                    print('len:', labels_len, out_len)
                    print('true labels, out (first 40 dims):', list(zip(labels_flat,out_flat))[:60])
                    print('accuracy:', accuracy_score(labels, out_flat))
                    print('majority class accuracy:', accuracy_score(labels, out_flat_zero))
                    
                    print('At step %i step-time %.4f loss %.4f' % (current_step, step_time, mean_train_loss))
                    print('Model saving path is:', model.out_dir)
                    print('Training started', (time.time() - training_start_time) / 3600.0,'hours ago.')
                    print('FLAGS params in short:',get_model_flags_param_short())

                    train_losses = []
                    step_time = 0
                    if checkpoint_step % FLAGS.checkpoints_per_save == 0:
                        min_loss = 1e10
                        if len(previous_losses) > 0:
                            min_loss = min(previous_losses)
                        if mean_train_loss < min_loss:
                            print(('Train loss: %.6f' % mean_train_loss) + (' is smaller than previous best loss: %.6f' % min_loss) )
                            print('Saving the best model so far to ', model.out_dir, '...')
                            model.saver.save(sess, model.out_dir, global_step=model.global_step)
                            previous_losses.append(mean_train_loss)

def compute_spk2len(spk2utt):
    return {spk: len(spk2utt[spk]) for spk in spk2utt.keys()}

def test_sampling(utt_id_list, spk2utt=None, spk2len=None, num_speakers=0 ):
    print('Generating sample:')
    print('test_sampling:', spk2utt[0])
    input_window_1, input_window_2, labels = get_batch_k_samples(idlist=utt_id_list, window_length=FLAGS.window_length, 
                                                                       window_neg_length=FLAGS.window_neg_length, left_contexts=FLAGS.left_contexts,
                                                                       right_contexts=FLAGS.right_contexts, k=FLAGS.negative_samples, debug=True,
                                                                       spk2utt=spk2utt, spk2len=spk2len , num_speakers=num_speakers)
    
    batch_size = len(input_window_1)
    
    print('num_speakers:', num_speakers)
    print('batch_size:', batch_size)
    print('Now plotting sample.')
    
    plt.figure(0)
    
    task_length = 2*(FLAGS.right_contexts + FLAGS.left_contexts)
    
    complete_window_seq = []
    for i in xrange(FLAGS.left_contexts):
        complete_window_seq.append(input_window_2[i])
   
    complete_window_seq.append(input_window_1[0])
     
    for i in xrange(FLAGS.right_contexts):
        complete_window_seq.append(input_window_2[FLAGS.left_contexts+i])
    
    complete_window_seq = np.vstack(complete_window_seq)
    
    plt.imshow(complete_window_seq.T, interpolation=None, aspect='auto', origin='lower')
    
    plt.figure(1)
    
    f, axarr = plt.subplots(task_length, sharex=True)
    
    expect_shape = input_window_1[0].shape
    for i in range(len(input_window_1)):
        if input_window_1[i].shape != expect_shape:
            print('warning:',i,input_window_1[i].shape)
    
    expect_shape = input_window_1[1].shape
    for i in range(len(input_window_1)):
        if input_window_2[i].shape != expect_shape:
            print('warning:',i,input_window_2[i].shape)
    
    
    for i,ax in enumerate(axarr):
        print(input_window_1[i])
        print('input_window_1[i]:',input_window_1[i].shape)
        print('input_window_2[i]:',input_window_2[i].shape)
        combined_feats = np.vstack([input_window_1[i], input_window_2[i]])
        print('combined_feats shape:',combined_feats.shape)
        im=ax.imshow(combined_feats.T, interpolation=None, aspect='auto', origin='lower')
        #ax.matshow(combined_feats.T)  
        ax.axvline(x=len(input_window_1[i])-0.5, color='k', linestyle='--')
        
    #plt.colorbar(im,ax=axarr[-1])
    plt.show()
    

if __name__ == "__main__":
    FLAGS._parse_flags()
    print("\nParameters:")
    print(get_FLAGS_params_as_str())

    #print(list(zip(utt_ids, filelist)))

    print('continuing training in 10 seconds...')
    #time.sleep(10)

    spk2utt, spk2len, num_speakers = None,None,None

    if FLAGS.end_to_end:
        utt_ids, filelist = utils.loadIdFile(FLAGS.filelist, 3000000)
        print(utt_ids, filelist)
        
        if FLAGS.debug:
            filelist = filelist[:10]
        
        id2file = {}
    
        utt_id_list = []
        i = 0
        for utt_id, myfile in itertools.zip_longest(utt_ids,filelist):
            if utt_id is None:
                utt_id = "audio_%06i" % i
            print('Loading:',utt_id,myfile)
            utt_id_list.append(utt_id)
            signal, framerate = utils.getSignal(myfile)
            if framerate != 16000:
                print('Warning framerate != 16000:', framerate)
           
            if signal.dtype != 'float32':
                print('dytpe is not float32', signal.dtype)
                signal = signal.astype('float32')
                #convert and clip to -1.0 - 1.0 range
                signal /= 32768.0
                signal = np.fmax(-1.0,signal)
                signal = np.fmin(1.0,signal)
            
            training_data[utt_id] = signal
            
            id2file[utt_id] = myfile
            
            i += 1
    else:
        #Using Kaldi scp
        if FLAGS.filelist.endswith('.scp'):
            features, utt_id_list = kaldi_io.readScp(FLAGS.filelist, limit = 1000 if FLAGS.debug else np.inf)
        elif FLAGS.filelist.endswith('.ark'):
            features, utt_id_list = kaldi_io.readArk(FLAGS.filelist, limit = 1000 if FLAGS.debug else np.inf)
            
        print(utt_id_list)
          
        min_required_sampling_length = FLAGS.window_length + FLAGS.window_neg_length * (FLAGS.left_contexts + FLAGS.right_contexts)
        print('min_required_sampling_length is:', min_required_sampling_length)
        
        training_data = {key: value for (key, value) in zip(utt_id_list, features) if value.shape[0] > min_required_sampling_length}
    
    if FLAGS.spk2utt != '':
        print('Loading speaker information from ', FLAGS.spk2utt)
        spk2utt = utils.loadSpk2Utt(FLAGS.spk2utt)
        #print('spk2utt:',spk2utt)
        del_list = []
        # Removing unavailable uttids. Either deleted because they are too short, or if debug = True and the size of the trainingset is reduced, most utt_ids are not available.
        # This is important, because the sampler expects every utt_ids for every speaker to be in the training set.
        for spk in spk2utt.keys():
            #print(spk2utt[spk])
            spk2utt[spk] = [elem for elem in spk2utt[spk] if elem in training_data]
        
            if len(spk2utt[spk]) == 0:
                del_list.append(spk)
        for spk in del_list:
            del spk2utt[spk]
       
        spk2len = compute_spk2len(spk2utt)
        
        print('spk2len:',spk2len)
        
        num_speakers = len(spk2len.keys())
        
    if FLAGS.test_sampling:
        test_sampling(utt_id_list, spk2utt=spk2utt, spk2len=spk2len, num_speakers=num_speakers)
    if FLAGS.gen_feats:
        gen_feat(utt_id_list, filelist, feats_outputfile=FLAGS.output_feat_file, feats_format=FLAGS.output_feat_format, hop_size = FLAGS.genfeat_hopsize)
    else:
        train(utt_id_list, spk2utt=spk2utt, spk2len=spk2len, num_speakers=num_speakers)
