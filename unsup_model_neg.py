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


from tensorflow.python.ops import control_flow_ops

if sys.version_info[0] == 3:
    xrange = range

tf.flags.DEFINE_boolean("end_to_end", True, "Use end-to-end learning (Input is 1D). Otherwise input is 2D like FBANK or MFCC features.")

tf.flags.DEFINE_integer("window1_length", 768, "First window length, samples or frames") # 100+ ms @ 16kHz
tf.flags.DEFINE_integer("window2_length", 768, "Second window length, samples or frames") # 100+ ms @ 16kHz

tf.flags.DEFINE_integer("embedding_size", 256 , "Fully connected size at the end of the network.")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")

tf.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                                "How many training steps to do per checkpoint.")
tf.flags.DEFINE_integer("steps_per_summary", 100,
                                "How many training steps to do per checkpoint.")

tf.flags.DEFINE_integer("checkpoints_per_save", 1,
                                "How many checkpoints until saving the model.")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_float("learn_rate", 5e-4, "Learn rate for the optimizer")

tf.flags.DEFINE_boolean("debug", False, "E.g. Smaller training data size")

tf.flags.DEFINE_boolean("log_tensorboard", True, "Log training process if this is set to True.")

tf.flags.DEFINE_string("train_dir", "/srv/data/milde/unspeech_models/", "Training dir to resume training from. If empty, a new one will be created.")

FLAGS = tf.flags.FLAGS

training_data = {}

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
    Unsupervised learning with RAW speech signals. This model learns a speech representation by u
    using a negative sampling objective, where true contexts must be discrimnated from samples ones
    """
    
    def create_training_graphs(self, create_new_train_dir=True, clip_norm=True, max_grad_norm=5.0):
        self.train_op = slim.learning.create_train_op(self.cost, self.optimizer, global_step=self.global_step)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.update_ops:
            print('Will add update_ops dependency ...')
            updates = tf.group(*self.update_ops)
            cross_entropy = control_flow_ops.with_dependencies([updates], self.cost)
        
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
    
    def get_random_audiosample(self, window_size):
        filelist_size = len(filelist)
        
        random_file_num = int(math.floor(np.random.random_sample() * filelist_size))
        random_file = filelist[random_file_num]
        audio_len = audio_data.shape[0] - window_size_1 - window_size_2
        random_pos_num = int(math.floor(np.random.random_sample() * audio_len))
        
        return np.array(audio_data[random_pos_num:random_pos_num+window_size])
   

    # does a batch where one of the examples are two windows with consecutive signals and k randomly selected window_2s
    #, with a fixed window1
    def get_batch_k_samples(self, filelist, window_size_1, window_size_2, k=4):            
        window1_batch = []
        window2_batch = []
        labels = []
        
        for i in xrange(FLAGS.batch_size*k):
            if i%k==0: 
                combined_sample = self.get_random_audiosample(window_size_1+window_size_2)
                window1 = combined_sample[:window_size_1]
                window2 = combined_sample[window_size_1:]
                #assign label 1, if both windows are consecutive
                labels.append(1)
                
            else:
                window1 = self.get_random_audiosample(window_size_1)
                window2 = self.get_random_audiosample(window_size_2)
                #assign label 0, if both windows are randomly selected
                labels.append(0)
                
            window1_batch.append(window1)
            window2_batch.append(window2)

        return window1,window2,labels
     
    # similar to get_batch_k_samples, but with true_context_window2_probability we select either two neighbooring pairs or two random audio snippets
    def get_batch_randomized(self, filelist, window_size_1, window_size_2, true_context_window2_probability=0.5):            
        window1_batch = []
        window2_batch = []
        labels = []
        
        for i in xrange(FLAGS.batch_size):
            if np.random.random_sample() <= true_context_window2_probability: 
                combined_sample = self.get_random_audiosample(window_size_1+window_size_2)
                window1 = combined_sample[:window_size_1]
                window2 = combined_sample[window_size_1:]
                #assign label 1, if both windows are consecutive
                labels.append(1)
                
            else:
                window1 = self.get_random_audiosample(window_size_1)
                window2 = self.get_random_audiosample(window_size_2)
                #assign label 0, if both windows are randomly selected 
                labels.append(0)
                
            window1_batch.append(window1)
            window2_batch.append(window2)

        return window1,window2,labels
    
    
    def __init__(self, window_size_1, window_size_2, filter_sizes, num_filters, fc_size, dropout_keep_prob, train_files, is_training=True, cost_function='mse', create_new_train_dir=True, batch_size=128):

        self.train_files = train_files

        self.window_size_1 = window_size_1
        self.window_size_2 = window_size_2
        self.fc_size = fc_size

        # None -> automatically sets the dimension to batch_size
        # window 1 is fixed
        self.input_window_1 = tf.placeholder(tf.float32, [None, window_size_1], name="input_window_1")
        # window 2 is either consecutive, or randomly sampled
        self.input_window_2 = tf.placeholder(tf.float32, [None, window_size_2], name="input_window_2")
        
        with tf.variable_scope("unsupmodel"):
            with tf.variable_scope("embedding-transform"):
                for i,input_window in enumerate([self.input_window_1, self.input_window_2]):
                    if i > 0: tf.get_variable_scope().reuse_variables()
                    #input_reshaped = tf.reshape(self.input_x, [-1, 1, window_length, 1])
                    window_length = int(input_window.get_shape([1]))
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
    
    
                    flattened_size = int(pooled.get_shape()[1]*pooled.get_shape()[2]*pooled.get_shape()[3])
                    # Reshape conv2 output to fit fully connected layer input
                    self.flattened_pooled = tf.reshape(pooled, [-1, flattened_size])
                
                    with tf.variable_scope('visualization_embedding'):
                        flattened_pooled_normalized = utils.tensor_normalize_0_to_1(self.flattened_pooled)
                        tf.summary.image('learned_embedding', tf.reshape(flattened_pooled_normalized,[-1,1,flattened_size,1]), max_outputs=10)
    
                    print('flattened_pooled shape:',self.flattened_pooled.get_shape())
    
                    self.fc1 = self.fully_connected(self.flattened_pooled, flattened_size, fc_size, name='fc1', use_dropout=False) #is_training)
                    print('fc1 shape:',self.fc1.get_shape())
