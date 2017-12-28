#-*- coding:utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import csv
import os
import decimal

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'output_dir', './pig_result', 'result file')

tf.app.flags.DEFINE_boolean(
    'dropout_keep_prob', 1.0,
    'Dropout keep probability.')

FLAGS = tf.app.flags.FLAGS

ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    #num_classes = dataset.num_classes-FLAGS.labels_offset
    num_classes = None
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, name] = provider.get(['image', 'name'])

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, names = tf.train.batch(
        [image, name],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    #################### 
    num_classes=dataset.num_classes - FLAGS.labels_offset

    with tf.variable_scope('InceptionV4', [images], reuse=None) as scope:
      with slim.arg_scope([slim.batch_norm, slim.dropout],
                          is_training=True):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          net, end_points = network_fn(images, scope=scope)

          # Auxiliary Head logits
          if num_classes:
            with tf.variable_scope('AuxLogits'):
              # 17 x 17 x 1024
              aux_logits = tf.slice(end_points['Mixed_6h'], [0, 0, 0, 0], [FLAGS.batch_size, -1, -1, -1])
              aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                           padding='VALID',
                                           scope='AvgPool_1a_5x5')
              aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                       scope='Conv2d_1b_1x1')
              aux_logits = slim.conv2d(aux_logits, 768,
                                       aux_logits.get_shape()[1:3],
                                       padding='VALID', scope='Conv2d_2a')
              aux_logits = slim.flatten(aux_logits)
              aux_logits = slim.fully_connected(aux_logits, num_classes,
                                                activation_fn=None,
                                                scope='Aux_logits')
              end_points['AuxLogits'] = aux_logits

          # Final pooling and prediction
          # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
          # can be set to False to disable pooling here (as in resnet_*()).
          with tf.variable_scope('Logits'): 
            # 8 x 8 x 1536
            # Triple Channle layers to extract image property
            triple_conv_outputs = []
            for i in xrange(3):
              with tf.variable_scope("Channel_Cov_%s" % i):
                 channel_layer = slim.conv2d(net, 512, [3, 3], scope="Conv2d_3x3")
                 channel_layer = slim.conv2d(channel_layer, 512, [3, 1], scope="Conv2d_3x1")
                 channel_layer = slim.conv2d(channel_layer, 512, [1, 3], scope="Conv2d_1x3")
              
                 triple_conv_outputs.append(channel_layer)
            net = tf.concat(axis=3, values=triple_conv_outputs)
                         
            kernel_size = net.get_shape()[1:3]
            if kernel_size.is_fully_defined():
              net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                    scope='AvgPool_1a')
            else:
              net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                   name='global_pool')
            end_points['global_pool'] = net
           
            # 1 x 1 x 1536
            net = slim.dropout(net, FLAGS.dropout_keep_prob, scope='Dropout_1b')
            net = slim.flatten(net, scope='PreLogitsFlatten')
            end_points['PreLogitsFlatten'] = net
            # 1536
            logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='Logits')
            end_points['Logits'] = logits

    logits = tf.nn.softmax(logits, 1)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()
   

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Testing %s' % checkpoint_path)
    
    sess_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        if variables_to_restore is not None:
            saver = tf_saver.Saver(variables_to_restore)
            saver.restore(sess, checkpoint_path)
        else:
            tf.logging.error("Fail to load checkpoint: %s", os.path.basename(checkpoint_path))
        tf.logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        results = {}
        keys = set([])
        print(num_batches)
        for i in xrange(int(num_batches)):
            batch_name, batch_logit = sess.run([names, logits])
            for i, n in enumerate(batch_name):
                if n in keys:
                    print(n)
                elif n[:-4] == "1363":
                    print("dd")
                    print(n)
                else:
                    keys.add(n)
                for j, l in enumerate(batch_logit[i]):
                    #results.append([n[:-4], dataset.labels_to_names[j], str(round(l, 6))])
                    if int(n[:-4]) not in results:
                        results[int(n[:-4])] = {}
                    results[int(n[:-4])][int(dataset.labels_to_names[j])] = l
        
        output_file = os.path.join(FLAGS.output_dir, FLAGS.model_name+".csv")
        tf.logging.info('Saving test result to %s.', output_file)
        n_k = results.keys()
        n_k.sort()
        with open(output_file, 'wb') as write_f:
            w = csv.writer(write_f)
            for n in n_k:
                cls_k = results[n].keys()
                cls_k.sort()
                for c in cls_k:
                    w.writerow([n, c, float_to_str(results[n][c])])

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
