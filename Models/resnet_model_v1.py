# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Layers import layers as custom_layers

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    bn_trainable=True):
    """Performs a batch normalization followed by a ReLU.

    Args:
        inputs: `Tensor` of shape `[batch, ..., channels]`.
        is_training: `bool` for whether the model is training.
        relu: `bool` if False, omits the ReLU operation.
        init_zero: `bool` if True, initializes scale parameter of batch
                normalization with 0 instead of 1 (default).

    Returns:
        A normalized `Tensor` with the same `data_format`.
    """
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    axis = 3

    inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=is_training,
            trainable=bn_trainable,
            fused=True,
            gamma_initializer=gamma_initializer)

    if relu:
        inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: `Tensor` of size `[batch, height, width, channels]`.
        kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
                operations. Should be a positive integer.

    Returns:
        A padded `Tensor` of the same `data_format` with size either intact
        (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                           [pad_beg, pad_end], [0, 0]])

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
        inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
        filters: `int` number of filters in the convolution.
        kernel_size: `int` size of the kernel to be used in the convolution.
        strides: `int` strides of the convolution.

    Returns:
        A `Tensor` of shape `[batch, height_out, width_out,  filters]`.
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


def residual_block(inputs, filters, is_training, strides,
                   use_projection=False, alignment=None, tf_layers=False,
                   bn_trainable=True):
    """Standard building block for residual networks with BN after convolutions.

    Args:
        inputs: `Tensor` of size `[batch, height, width, channels]`.
        filters: `int` number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
        is_training: `bool` for whether the model is in training.
        strides: `int` block stride. If greater than 1, this block will ultimately
                downsample the input.
        use_projection: `bool` for whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.

    Returns:
        The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
        # Projection shortcut in first layer to match filters and strides
        with tf.variable_scope("conv0"):
            use_relu = False
            shortcut = conv2d_fixed_padding(
                    inputs=inputs, filters=filters, kernel_size=1,
                    strides=strides, alignment=alignment, alignment_relu=use_relu,
                    tf_layers=tf_layers)
            shortcut = batch_norm_relu(shortcut, is_training, relu=use_relu,
                                       bn_trainable=bn_trainable)
    with tf.variable_scope("conv1"):
        use_relu = True
        inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                alignment=alignment, alignment_relu=use_relu,
                tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                 bn_trainable=bn_trainable)

    with tf.variable_scope("conv2"):
        use_relu = False
        inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=1,
                alignment=alignment, alignment_relu=use_relu,
                tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                 init_zero=True, bn_trainable=bn_trainable)

    return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs, filters, is_training, strides,
                     use_projection=False, alignment=None, tf_layers=False,
                     bn_trainable=True):
    """Bottleneck block variant for residual networks with BN after convolutions.

    Args:
        inputs: `Tensor` of size `[batch, height, width, channels]`.
        filters: `int` number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
        is_training: `bool` for whether the model is in training.
        strides: `int` block stride. If greater than 1, this block will ultimately
                downsample the input.
        use_projection: `bool` for whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.

    Returns:
        The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
        # Projection shortcut only in first block within a group. Bottleneck blocks
        # end with 4 times the number of filters.
        with tf.variable_scope("conv0"):
            filters_out = 4 * filters
            use_relu = False
            shortcut = conv2d_fixed_padding(
                    inputs=inputs, filters=filters_out, kernel_size=1,
                    strides=strides, alignment=alignment, alignment_relu=use_relu,
                    tf_layers=tf_layers)
            shortcut = batch_norm_relu(shortcut, is_training, relu=use_relu,
                                       bn_trainable=bn_trainable)

    with tf.variable_scope("conv1"):
        use_relu = True
        inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1,
                alignment=alignment, alignment_relu=use_relu,
                tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                 bn_trainable=bn_trainable)

    with tf.variable_scope("conv2"):
        use_relu = True
        inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                alignment=alignment, alignment_relu=use_relu,
                tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                 bn_trainable=bn_trainable)

    with tf.variable_scope("conv3"):
        use_relu = False
        inputs = conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
                alignment=alignment, alignment_relu=use_relu,
                tf_layers=tf_layers)
        inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                 init_zero=True, bn_trainable=bn_trainable)

    return tf.nn.relu(inputs + shortcut)


def block_group(inputs, filters, block_fn, blocks, strides, is_training, name,
                alignment=None, tf_layers=False, endpoints=None,
                bn_trainable=True):
    """Creates one group of blocks for the ResNet model.

    Args:
        inputs: `Tensor` of size `[batch, height, width, channels]`.
        filters: `int` number of filters for the first convolution of the layer.
        block_fn: `function` for the block to use within the model
        blocks: `int` number of blocks contained in the layer.
        strides: `int` stride to use for the first convolution of the layer. If
                greater than 1, this layer will downsample the input.
        is_training: `bool` for whether the model is training.
        name: `str`name for the Tensor output of the block layer.

    Returns:
        The output `Tensor` of the block layer.
    """
    # Only the first block per block_group uses projection shortcut and strides.
    with tf.variable_scope(name + "_{}".format(0)):
        inputs = block_fn(inputs, filters, is_training, strides,
                          use_projection=True, alignment=alignment,
                          tf_layers=tf_layers, bn_trainable=bn_trainable)

        if endpoints is not None:
            endpoints[name + '_' + str(0)] = inputs

    for i in range(1, blocks):
        with tf.variable_scope(name + "_{}".format(i)):
            inputs = block_fn(inputs, filters, is_training, 1,
                              alignment=alignment, tf_layers=tf_layers,
                              bn_trainable=bn_trainable)

            if endpoints is not None:
                endpoints[name + '_' + str(i)] = inputs

    return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, num_classes,
                        alignment=None, tf_layers=False, bn_trainable=True):
    """Generator for ResNet v1 models.

    Args:
        block_fn: `function` for the block to use within the model. Either
                `residual_block` or `bottleneck_block`.
        layers: list of 4 `int`s denoting the number of blocks to include in each
            of the 4 block groups. Each group consists of blocks that take inputs of
            the same resolution.
        num_classes: `int` number of possible classes for image classification.

    Returns:
        Model `function` that takes in `inputs` and `is_training` and returns the
        output `Tensor` of the ResNet model.
    """
    def model(inputs, is_training, return_endpoints=False):
        """Creation of the model graph."""
        print(inputs.shape.as_list())
        input_width = inputs.shape.as_list()[2]

        endpoints = {}
        print("initial shape", inputs.name, inputs.shape)
        with tf.variable_scope("conv0"):
            use_relu = True
            inputs = conv2d_fixed_padding(
                    inputs=inputs, filters=64, kernel_size=7, strides=2,
                    alignment=alignment, alignment_relu=use_relu,
                    tf_layers=tf_layers)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = batch_norm_relu(inputs, is_training, relu=use_relu,
                                     bn_trainable=bn_trainable)

        inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=3, strides=2, padding='SAME')
        inputs = tf.identity(inputs, 'initial_max_pool')
        endpoints['block_group0'] = inputs
        inputs = block_group(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=1, is_training=is_training, name='block_group1',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_group1'] = inputs
        inputs = block_group(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='block_group2',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_group2'] = inputs
        inputs = block_group(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=2, is_training=is_training, name='block_group3',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_group3'] = inputs
        inputs = block_group(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=2, is_training=is_training, name='block_group4',
                alignment=alignment, tf_layers=tf_layers, endpoints=endpoints,
                bn_trainable=bn_trainable)
        endpoints['block_group4'] = inputs

        if input_width == 224:
            final_pool_size = 7
        elif input_width == 128:
            final_pool_size = 4
        else:
            raise NotImplementedError("Image input must be size 224 or 128!")

        print(inputs.name, inputs.shape)
        # The activation is 7x7 so this is a global average pool.
        inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=final_pool_size, strides=1, padding='VALID')
        inputs = tf.identity(inputs, 'final_avg_pool')
        endpoints['final_avg_pool'] = inputs
        inputs = tf.reshape(
                inputs, [-1, 2048 if block_fn is bottleneck_block else 512])
        with tf.variable_scope("dense"):
            if tf_layers:
                inputs = tf.layers.dense(
                        inputs=inputs,
                        units=num_classes,
                        kernel_initializer=tf.random_normal_initializer(stddev=.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=None)
            else:
                inputs = custom_layers.Dense(
                        input=inputs,
                        units=num_classes,
                        kernel_initializer=tf.random_normal_initializer(stddev=.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        alignment=alignment,
                        activation=None)
        inputs = tf.identity(inputs, 'final_dense')
        endpoints['final_dense'] = inputs

        if return_endpoints:
            return inputs, endpoints
        else:
            return inputs

    model.default_image_size = 224
    return model


def resnet_v1(resnet_depth, num_classes, alignment=None, tf_layers=False,
              bn_trainable=True):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    print("Setting batch_normalization trainable to {}".format(bn_trainable))
    params = model_params[resnet_depth]
    return resnet_v1_generator(
            params['block'], params['layers'], num_classes,
            alignment=alignment, tf_layers=tf_layers, bn_trainable=bn_trainable)
