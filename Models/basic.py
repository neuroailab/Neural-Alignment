import tensorflow as tf
from Layers import layers

# TODO: parametrize this fully
def fc(inputs,
       num_classes=10,
       layers_list=[],
       activation=None,
       alignment=None,
       **kwargs):
    # I think this is only on GPU. This next line might be differerent on TPU
    im = inputs['images']

    # Hidden layers
    for index, units in enumerate(layers_list):
        with tf.variable_scope("dense{}".format(index)):
            im = layers.Dense(im,
                              units=units,
                              alignment=alignment,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6),
                              activation=activation)

    # Output layer
    with tf.variable_scope("dense_output"):
        logits = layers.Dense(im,
                              units=num_classes,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6),
                              alignment=alignment)

    # Recall the interface here is also different for TPU
    return logits, {}


# TODO: parametrize this fully
def conv(inputs,
         num_classes=10,
         filters_list=[],
         kernels_list=[],
         alignment=None,
         **kwargs):

    im = inputs['images']

    # Hidden layers
    im = tf.reshape(im, [-1,28,28,1])
    for index, (filters, kernel_size) in enumerate(zip(filters_list, kernels_list)):
        with tf.variable_scope("conv{}".format(index)):
            im = layers.Conv2D(im,
                               filters,
                               kernel_size,
                               alignment=alignment,
                               activation="sigmoid")

    # Output layer
    im = tf.layers.flatten(im)
    with tf.variable_scope("dense_output"):
        logits = layers.Dense(im,
                              units=num_classes,
                              alignment=alignment)

    return logits, {}
