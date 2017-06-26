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

tf.flags.DEFINE_string("filelist", "filelist.english.train", "Filelist, one wav file per line")
tf.flags.DEFINE_boolean("end_to_end", True, "Use end-to-end learning (Input is 1D). Otherwise input is 2D like FBANK or MFCC features.")
tf.flags.DEFINE_boolean("debug", True, "Limits the filelist size and is more debug.")

tf.flags.DEFINE_integer("sample_rate", 16000, "Sample rate of the audio files. Must have the same samplerate for all audio files.") # 100+ ms @ 16kHz
tf.flags.DEFINE_string("filter_sizes", "512", "Comma-separated filter sizes (default: '200')") # 25ms @ 16kHz
tf.flags.DEFINE_integer("num_filters", 40, "Number of filters per filter size (default: 40)")
tf.flags.DEFINE_integer("window1_length", 1024, "First window length, samples or frames") # 100+ ms @ 16kHz
tf.flags.DEFINE_integer("window2_length", 1024, "Second window length, samples or frames") # 100+ ms @ 16kHz
tf.flags.DEFINE_integer("embedding_size", 256 , "Fully connected size at the end of the network.")

tf.flags.DEFINE_boolean("with_vgg16", True, "Whether to use a vgg16 network for the embeddings computation.")
tf.flags.DEFINE_boolean("with_dense_network", False,  "Whether to use a dense conv network for the embeddings computation.")

tf.flags.DEFINE_integer("dense_block_filters", 5,  "Number of filters inside a conv2d in a dense block.")
tf.flags.DEFINE_integer("dense_block_layers_connected", 3,  "Number of layers inside dense block.")
tf.flags.DEFINE_integer("dense_block_filters_transition", 4, "Number of filters inside a conv2d in a dense block transition.")

tf.flags.DEFINE_boolean("tied_embeddings_transforms", True, "Whether the transformations of the embeddings windows should have tied weights. Only makes sense if the window sizes match.")

tf.flags.DEFINE_integer("negative_samples", 2, "How many negative samples to generate.")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
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

tf.flags.DEFINE_float("learn_rate", 5e-4, "Learn rate for the optimizer")
tf.flags.DEFINE_float("gradient_clipping", 5.0, "Clip the gradient at larger +/- this value.")

tf.flags.DEFINE_boolean("log_tensorboard", True, "Log training process if this is set to True.")

tf.flags.DEFINE_string("train_dir", "/srv/data/milde/unspeech_models/neg/", "Training dir to resume training from. If empty, a new one will be created.")

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
        a = slim.conv2d(input_layer,filters,[3,3], activation_fn=non_linearity, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer = tf.constant_initializer(0.01))
        nodes.append(a)
        for z in range(num_connected):
            b = slim.conv2d(tf.concat(nodes,3),filters,[3,3], activation_fn=non_linearity, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer = tf.constant_initializer(0.01))
            nodes.append(b)
        return b

#https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py
def DenseTransition2D(l, filters, name, with_conv=True, non_linearity=lrelu):
    with tf.variable_scope(name):
        if with_conv:
            l = slim.conv2d(l,filters,[3,3], activation_fn=non_linearity, weights_initializer=tf.contrib.layers.variance_scaling_initializer(), biases_initializer = tf.constant_initializer(0.01))
        l = slim.avg_pool2d(l, [2,2])
    return l

def DenseFinal2D(l, name, pool_size=7):
    with tf.variable_scope(name):
        l = slim.avg_pool2d(l, [pool_size,pool_size], stride=1)
    return l

#from https://github.com/tensorflow/tensorflow/tree/r1.2/tensorflow/contrib/slim
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=lrelu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      biases_initializer = tf.constant_initializer(0.01)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 1024, scope='fc6')
    #net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 256, scope='fc7')
    #net = slim.dropout(net, 0.5, scope='dropout7')
    #net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net

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
    
    def get_random_audiosample(self, window_size):
        filelist_size = len(filelist)
        
        random_file_num = int(math.floor(np.random.random_sample() * filelist_size))
        random_file = filelist[random_file_num]
        audio_data = training_data[random_file]
        audio_len = audio_data.shape[0] - window_size
        random_pos_num = int(math.floor(np.random.random_sample() * audio_len))
        
        return np.array(audio_data[random_pos_num:random_pos_num+window_size])
   

    # does a batch where one of the examples are two windows with consecutive signals and k randomly selected window_2s
    #, with a fixed window1
    def get_batch_k_samples(self, filelist, window_size_1, window_size_2, k=4):            
        window1_batch = []
        window2_batch = []
        labels = []
        
        for i in xrange(FLAGS.batch_size*(k+1)):
            if i%(k+1)==0: 
                combined_sample = self.get_random_audiosample(window_size_1+window_size_2)
                window1 = combined_sample[:window_size_1]
                window2 = combined_sample[window_size_1:]
                #assign label 1, if both windows are consecutive
                labels.append(1.0)
                
            else:
                window1 = self.get_random_audiosample(window_size_1)
                window2 = self.get_random_audiosample(window_size_2)
                #assign label 0, if both windows are randomly selected
                labels.append(0.0)
                
            window1_batch.append(window1)
            window2_batch.append(window2)

        labels = np.asarray(labels).reshape(-1,1)

        #if self.first_call_to_get_batch:
        #    print("window1_batch,",[elem[:5] for elem in window1_batch],"window2_batch,",[elem[:5] for elem in window2_batch],"labels",labels) 
        #    self.first_call_to_get_batch = False

        return window1_batch,window2_batch,labels
     
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
                labels.append(1.0)
                
            else:
                window1 = self.get_random_audiosample(window_size_1)
                window2 = self.get_random_audiosample(window_size_2)
                #assign label 0, if both windows are randomly selected 
                labels.append(0.0)
                
            window1_batch.append(window1)
            window2_batch.append(window2)

        return window1_batch,window2_batch,labels
    
    
    def __init__(self, window_size_1, window_size_2, filter_sizes, num_filters, fc_size, dropout_keep_prob, train_files, is_training=True, create_new_train_dir=True, batch_size=128):

        self.train_files = train_files

        self.window_size_1 = window_size_1
        self.window_size_2 = window_size_2
        self.fc_size = fc_size

        # None -> automatically sets the dimension to batch_size
        # window 1 is fixed
        self.input_window_1 = tf.placeholder(tf.float32, [None, window_size_1], name="input_window_1")
        # window 2 is either consecutive, or randomly sampled
        self.input_window_2 = tf.placeholder(tf.float32, [None, window_size_2], name="input_window_2")
        
        self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        
        self.first_call_to_get_batch = True
        
        with tf.variable_scope("unsupmodel"):
            # a list of embeddings to use for the binary classifier (the embeddings are combined)
            self.outs = []
            with tf.variable_scope("embedding-transform"):
                for i,input_window in enumerate([self.input_window_1, self.input_window_2]):
                    if FLAGS.tied_embeddings_transforms and i > 0: 
                        print("Reusing variables for embeddings computation.")
                        tf.get_variable_scope().reuse_variables()
                    #input_reshaped = tf.reshape(self.input_x, [-1, 1, window_length, 1])
                    window_length = int(input_window.get_shape()[1])
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
                    conv = lrelu(tf.nn.bias_add(conv, b), name="activation1")
    
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
        
                    needs_flattening = True
                    if FLAGS.with_dense_network:
                        
                        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm if FLAGS.batch_normalization else None,
                                                    normalizer_params={'is_training': is_training, 'decay': 0.95} if FLAGS.batch_normalization else None):
                            
                            #input_layer,filters, layer_num, num_connected, non_linearity=lrelu
                            conv = DenseBlock2D(input_layer=pooled, filters=FLAGS.dense_block_filters, layer_num=2, num_connected=FLAGS.dense_block_layers_connected) #tf.nn.conv2d(pooled, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                            pooled = DenseTransition2D(l=conv, filters=FLAGS.dense_block_filters_transition, name='transition1', with_conv=True) 
                            
                            conv = DenseBlock2D(pooled, filters=FLAGS.dense_block_filters, layer_num=3, num_connected=FLAGS.dense_block_layers_connected)
                            #pooled = DenseTransition2D(conv, 40, 'transition2')
                            pooled = DenseFinal2D(conv, 'dense_end')
    
                        print('pool shape after dense blocks:', pooled.get_shape())
    
                    if FLAGS.with_vgg16:
                        pooled = vgg16(pooled)
                        print('pool shape after vgg16 block:', pooled.get_shape())
    
                    if needs_flattening:
                        flattened_size = int(pooled.get_shape()[1]*pooled.get_shape()[2]*pooled.get_shape()[3])
                        # Reshape conv2 output to fit fully connected layer input
                        self.flattened_pooled = tf.reshape(pooled, [-1, flattened_size])
                    else:
                        self.flattened_pooled = pooled
                
                    #with tf.variable_scope('visualization_embedding'):
                    #    flattened_pooled_normalized = utils.tensor_normalize_0_to_1(self.flattened_pooled)
                    #    tf.summary.image('learned_embedding', tf.reshape(flattened_pooled_normalized,[-1,1,flattened_size,1]), max_outputs=10)
    
                    print('flattened_pooled shape:',self.flattened_pooled.get_shape())
    
                    self.fc1 = slim.fully_connected(self.flattened_pooled, fc_size, activation_fn=lrelu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)) #is_training)
                    print('fc1 shape:',self.fc1.get_shape())
                    self.outs.append(self.fc1)
                    
            stacked = tf.concat(self.outs, 1)
            print('stacked shape:',stacked.get_shape())
                
            self.out = slim.fully_connected(stacked, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.out))
    
            if is_training:
                self.create_training_graphs(create_new_train_dir)
                self.saver = tf.train.Saver(tf.global_variables())

    # do a training step with the supplied input data
    def step(self, sess, input_window_1, input_window_2, labels):
        feed_dict = {self.input_window_1: input_window_1, self.input_window_2: input_window_2, self.labels: labels}
        _, output, loss = sess.run([self.train_op, self.out, self.cost], feed_dict=feed_dict)
        return  output, loss
    
def train(filelist):
    filter_sizes = [int(x) for x in FLAGS.filter_sizes.split(',')]
    with tf.device('/gpu:1'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            model = UnsupSeech(window_size_1=FLAGS.window1_length, window_size_2=FLAGS.window2_length, filter_sizes=filter_sizes, 
                                num_filters=FLAGS.num_filters, fc_size=FLAGS.embedding_size, dropout_keep_prob=FLAGS.dropout_keep_prob, train_files = filelist,  batch_size=FLAGS.batch_size)
            
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
                    #input_window_1, input_window_2, labels = model.get_batch_k_samples(filelist=filelist, window_size_1=FLAGS.window1_length, window_size_2=FLAGS.window2_length, k=FLAGS.negative_samples)
                    summary_str = sess.run(model.train_summary_op, feed_dict={model.input_window_1:input_window_1, model.input_window_2:input_window_2, model.labels: labels})
                    summary_writer.add_summary(summary_str, current_step)

                # Get a batch and make a step.
                start_time = time.time()
                input_window_1, input_window_2, labels = model.get_batch_k_samples(filelist=filelist, window_size_1=FLAGS.window1_length, window_size_2=FLAGS.window2_length, k=FLAGS.negative_samples)
                out, train_loss = model.step(sess, input_window_1, input_window_2, labels)
                train_losses.append(train_loss)

                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    checkpoint_step += 1
                    mean_train_loss = np.mean(train_losses)

                    #print('input_window_1:', input_window_1[0])
                    #print('input_window_2:', input_window_2[0])
                    print('true labels, out (first 40 dims):', list(zip([elem[0] for elem in labels[:40]],[1.0 if elem[0] > 0.5 else 0.0 for elem in out[:40]])))
                    print('At step %i step-time %.4f loss %.4f' % (current_step, step_time, mean_train_loss))
                    
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


if __name__ == "__main__":
    FLAGS._parse_flags()
    print("\nParameters:")
    print(get_FLAGS_params_as_str())
    filelist = utils.loadIdFile(FLAGS.filelist, 3000000)
    print(filelist)

    print('continuing training in 5 seconds...')
    time.sleep(5)

    if FLAGS.debug:
        filelist = filelist[:5]

    for myfile in filelist:
#    for myfile in [filelist[-1]]:   
        print('Loading:',myfile)
        signal = np.float32(utils.getSignal(myfile)[0])
        #convert and clip to -1.0 - 1.0 range
        signal /= 32768.0
        signal = np.fmax(-1.0,signal)
        signal = np.fmin(1.0,signal)
        
        training_data[myfile] = signal
    
    #todo add eval and writing out features
    train(filelist)
