# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Most code in this file was borrowed from https://github.com/anishathalye/neural-style/blob/master/vgg.py
# Most code in this file was borrowed from https://github.com/cindydeng1991/Wavelet-Domain-Style-Transfer-for-an-Effective-Perception-distortion-Tradeoff-in-Single-Image-Super-
# model download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
import numpy as np
from scipy.io import loadmat


class Vgg19(object):
    def __init__(self, data_path):
        data = loadmat(data_path)
        self.mean_pixel = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        self.weights = data['layers'][0]
        self.layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')
        self.tf_weights = {}
        for i, name in enumerate(self.layers):
            if name[:4] == 'conv':
                with tf.variable_scope('VGG19'):
                    kernels = self.weights[i][0][0][2][0][0]
                    bias = self.weights[i][0][0][2][0][1]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = tf.constant(np.transpose(kernels, (1, 0, 2, 3)))
                    bias = tf.constant(bias.reshape(-1))
                self.tf_weights[name] = kernels
                self.tf_weights[name+'b'] = bias

    def build(self, input, scope='VGGModel'):
        layers = {}
        current = input * 255.0 - self.mean_pixel

        with tf.name_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels = self.tf_weights[name]
                    bias = self.tf_weights[name+'b']
                    current = tf.nn.bias_add(tf.nn.conv2d(current, kernels, strides=(1, 1, 1, 1), padding='SAME', name=name), bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current, name=name)
                elif kind == 'pool':
                    current = tf.nn.max_pool2d(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name=name)
                layers[name] = current

        assert len(layers) == len(self.layers)
        return layers
