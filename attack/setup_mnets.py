## setup_mnets.py -- sets up mobile nets
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request
import setup_inception
import sys
sys.path.append(os.path.abspath('../models/research/slim'))
from nets import nets_factory as factory
from datasets import dataset_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               lookup_path=None):
    if not lookup_path:
      lookup_path = os.path.join(
          '/data/pca/data/imagenet_final/', 'labels.txt')
    self.node_lookup = self.load(lookup_path)

  def load(self, lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """

    # Loads mapping from string UID to human-readable string
    node_id_to_name = {}
    proto_as_ascii = open(lookup_path).readlines()
    for line in proto_as_ascii:
        target_class = int(line.split(':')[0])
        target_class_string = line.split(':')[1]
        node_id_to_name[target_class] = target_class_string

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


class ImagenetTF:
    def __init__(self, model_name, eval_image_size):
        self.model_name = model_name
        self.dataset = dataset_factory.get_dataset(
            "imagenet", "validation", "/data/pca/data/imagenet_final")
        batch_size = 100

        provider = slim.dataset_data_provider.DatasetDataProvider(
            self.dataset,
            shuffle=False,
            common_queue_capacity=batch_size * 2,
            common_queue_min=batch_size * 2)
        
        [image, label] = provider.get(['image', 'label'])
        
        preprocessing_name = "mobilenet_v1" if self.model_name.startswith("mobilenet") else self.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=5 * batch_size)

        self.images = images
        self.labels = labels
        
    def get_batch(self, sess):
        test_data, test_labels = sess.run([self.images, self.labels])
        self.test_data = test_data
        self.test_labels = np.zeros((len(test_labels), 1001))
        self.test_labels[np.arange(len(test_labels)), test_labels] = 1

    def get_data(self, sess):
        d, l = sess.run([self.images, self.labels])
        return (d, l)

class MNETSModel:
    def __init__(self, ckpt_path, model_name, batch_size, session):
        self.num_labels = 1001
        self.num_channels = 3

        self.sess = session
        self.ckpt_path = ckpt_path
        self.model_name = model_name

        self.model = factory.get_network_fn(self.model_name,
                                            num_classes=self.num_labels,
                                            is_training=False)

        self.image_size = self.model.default_image_size
        self.batch_size = batch_size
        self.reuse = False

    def predict(self, data, reuse=False):
        self.logits, _ = self.model(data, reuse=self.reuse)
        if not self.reuse:
            self.reuse = True
        return self.logits
        
        
    
