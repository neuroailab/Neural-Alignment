# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Layers import layers as custom_layers

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, bn_trainable=True):
    """Performs a batch normalization followed by a ReLU.

    Args:
      inputs: A Tensor of shape [batch, ..., channels].
      is_training: Whether the model is currently training.

    Returns:
      A Tensor of shape [batch, ..., channels]
    """
    # We set fused=True for a significant performance boost.
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        trainable=bn_trainable,
        fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A Tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.

    Returns:
      A Tensor of size [batch, height_out, width_out, channels] with the
        input either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                           [pad_beg, pad_end], [0,0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         alignment=None,
                         alignment_relu=False,
                         alignment_batchnorm=True,
                         tf_layers=False):
    """Strided 2-D convolution with explicit padding.

    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

    Args:
      inputs: A Tensor of size [batch, height_in, width_in, channels].
      filters: The number of filters in the convolution.
      kernel_size: The size of the kernel to be used in the convolution.
      strides: The strides of the convolution.

    Returns:
      A Tensor of shape [batch, height_out, width_out, filters].
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    if tf_layers:
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
    else:
        return custom_layers.Conv2D(
            input=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            alignment=alignment,
            alignment_relu=alignment_relu,
            alignment_batchnorm=alignment_batchnorm)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   alignment=None, tf_layers=False, bn_trainable=True):
    """Standard building block for residual networks with BN before convolutions.

    Args:
    inputs: A Tensor of size [batch, height, width, channels].
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.

    Returns:
    The output Tensor of the block.
    """
    shortcut = inputs
    with tf.variable_scope("bn0"):
        inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        with tf.variable_scope("conv0"):
            shortcut = projection_shortcut(inputs)

    with tf.variable_scope("conv1"):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            alignment=alignment, alignment_relu=True, tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

    with tf.variable_scope("conv2"):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            alignment=alignment, tf_layers=tf_layers)

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, alignment=None, tf_layers=False,
                     bn_trainable=True):
    """Bottleneck block variant for residual networks with BN before convolutions.

    Args:
    inputs: A Tensor of size [batch, height, width, channels].
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.

    Returns:
    The output Tensor of the block.
    """
    shortcut = inputs
    with tf.variable_scope("bn0"):
        inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        with tf.variable_scope("conv0"):
            shortcut = projection_shortcut(inputs)

    with tf.variable_scope("conv1"):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            alignment=alignment, alignment_relu=True, tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

    with tf.variable_scope("conv2"):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            alignment=alignment, alignment_relu=True, tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

    with tf.variable_scope("conv3"):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            alignment=alignment, tf_layers=tf_layers)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                alignment=None, tf_layers=False, endpoints=None,
                bn_trainable=True):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A Tensor of size [batch, height, width, channels].
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the Tensor output of the block layer.

    Returns:
    The output Tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            alignment=alignment, tf_layers=tf_layers)

    # Only the first block per block_layer uses projection_shortcut and strides
    with tf.variable_scope(name + "_{}".format(0)):
        inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                          alignment=alignment, tf_layers=tf_layers,
                          bn_trainable=bn_trainable)
        if endpoints is not None:
            endpoints[name + '_' + str(0)] = inputs

    for i in range(1, blocks):
        with tf.variable_scope(name + "_{}".format(i)):
            inputs = block_fn(inputs, filters, is_training, None, 1,
                              alignment=alignment, tf_layers=tf_layers,
                              bn_trainable=bn_trainable)
            if endpoints is not None:
                endpoints[name + '_' + str(i)] = inputs

    return tf.identity(inputs, name)


def resnet_v2_generator(block_fn, layers, num_classes,
                        alignment=None, tf_layers=False, bn_trainable=True):
    """Generator for ResNet v2 models.

    Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.

    Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output Tensor of the ResNet model.
    """
    def model(inputs, is_training, return_endpoints=False):
        endpoints = {}
        input_width = inputs.shape.as_list()[2]

        with tf.variable_scope("conv0"):
            inputs = conv2d_fixed_padding(
                    inputs=inputs, filters=64, kernel_size=7, strides=2,
                    alignment=alignment, tf_layers=tf_layers)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=3, strides=2, padding='SAME',
                data_format='channels_last')
        inputs = tf.identity(inputs, 'initial_max_pool')
        endpoints['block_layer0'] = inputs
        inputs = block_layer(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=1, is_training=is_training, name='block_layer1',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_layer1'] = inputs
        inputs = block_layer(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='block_layer2',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_layer2'] = inputs
        inputs = block_layer(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=2, is_training=is_training, name='block_layer3',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_layer3'] = inputs
        inputs = block_layer(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=2, is_training=is_training, name='block_layer4',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_layer4'] = inputs

        with tf.variable_scope("bnl4"):
            inputs = batch_norm_relu(inputs, is_training, bn_trainable=bn_trainable)

        if input_width == 299:
            final_pool_size = 10
        elif input_width == 224:
            final_pool_size = 7
        elif input_width == 128:
            final_pool_size = 4
        else:
            raise NotImplementedError("Image input must be size 224 or 128!")
        inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=final_pool_size, strides=1, padding='VALID',
                data_format='channels_last')
        inputs = tf.identity(inputs, 'final_avg_pool')
        endpoints['final_avg_pool'] = inputs
        inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
        print("final tensors before fc")
        print(inputs.name, inputs.shape.as_list())
        with tf.variable_scope("dense"):
            if tf_layers:
                inputs = tf.layers.dense(
                        inputs=inputs,
                        units=num_classes,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=None)
            else:
                inputs = custom_layers.Dense(
                        input=inputs,
                        units=num_classes,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        alignment=alignment,
                        activation=None)
        inputs = tf.identity(inputs, 'final_dense')
        print("logits name", inputs.name)
        endpoints['final_dense'] = inputs

        if return_endpoints:
            return inputs, endpoints
        else:
            return inputs

    model.default_image_size = 224
    return model


def resnet_v2(resnet_size, num_classes, alignment=None, tf_layers=False,
              bn_trainable=True):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    print("Setting batch_normalization trainable to {}".format(bn_trainable))
    params = model_params[resnet_size]
    return resnet_v2_generator(params['block'],
                               params['layers'],
                               num_classes,
                               alignment=alignment, tf_layers=tf_layers,
                               bn_trainable=bn_trainable)

