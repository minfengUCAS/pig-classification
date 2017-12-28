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
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

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
    'output_dir', './validation_result', 'result file')

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
    if FLAGS.model_name[:] == "resnet":
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
    [image, label, name] = provider.get(['image', 'label','name'])
    label -= FLAGS.labels_offset
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images,labels, names = tf.train.batch(
        [image,label,name],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    net, end_points = network_fn(images)
    num_classes=dataset.num_classes - FLAGS.labels_offset

    with tf.variable_scope('InceptionV4', [net], reuse=None) as scope:
      with slim.arg_scope([slim.batch_norm, slim.dropout],
                          is_training=True):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
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
            # 1 x 1 x 1536
            net = slim.dropout(net, FLAGS.dropout_keep_prob, scope='Dropout_1b')
            net = slim.flatten(net, scope='PreLogitsFlatten')
            end_points['PreLogitsFlatten'] = net
            # 1536
            logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='Logits')
    #####################
    # Add dropout layer #
    #####################
    if FLAGS.model_name[:6] == "resnet":
      with tf.variable_scope(FLAGS.model_name, 'my_logits', [logits], reuse=None) as scope:
        with slim.arg_scope([slim.dropout], is_training=True):
          logits = slim.dropout(logits, FLAGS.dropout_keep_prob, scope='my_dropout')
          #logits = tf.reshape(logits, [-1, FLAGS.feature_size*FLAGS.feature_size*FLAGS.channel_num])
          logits = slim.flatten(logits, scope='flatten')
          logits = slim.fully_connected(logits, dataset.num_classes-FLAGS.labels_offset, 
                                        activation_fn=None, scope='my_logits_layers')
          end_points['My_Logits'] = logits
    logits = tf.nn.softmax(logits, 1)
    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()
   
    '''
    for var in variables_to_restore:
      tf.logging.info('%s' % var.op.name)
    '''

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
        results_pred_label = []
        all_label = []
        all_pred = []
        log_loss = 0.0
        num = 0
        keys = set([])
        print(num_batches)
        for num_batch in xrange(int(num_batches)):
            batch_name, batch_logit, batch_prediction, batch_label = sess.run([names, logits, predictions, labels])
            all_label = np.concatenate([all_label, batch_label])
            all_pred = np.concatenate([all_pred, batch_prediction])
            for i, n in enumerate(batch_name):
                if n in keys:
                    print(n)
                else:
                    keys.add(n)
                
                if batch_prediction[i] != batch_label[i]:
                    results_pred_label.append([n, batch_prediction[i], batch_label[i]])

                for j, l in enumerate(batch_logit[i]):
                    #results.append([n[:-4], dataset.labels_to_names[j], str(round(l, 6))])
                    if n not in results:
                        results[n] = {}
                    if j == batch_label[i]:
                        log_loss -= math.log(l)
                        num += 1

                    results[n][int(dataset.labels_to_names[j])] = l
        
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

        output_file = os.path.join(FLAGS.output_dir, FLAGS.model_name+"_err.csv")
        with open(output_file, 'wb') as write_f:
            w = csv.writer(write_f)
            for line in results_pred_label:
                w.writerow(line)
        print(log_loss)
        print(num)
        print(float(log_loss)/float(num))
        print()
        correct_predictions = float(sum(all_label == all_pred))
        print("Total number of example: {}".format(len(all_pred)))
        print("Correct number of example: {}".format(correct_predictions))
        print("Accuracy: {:g}".format(correct_predictions/float(len(all_pred))))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
