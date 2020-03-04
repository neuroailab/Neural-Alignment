import tensorflow as tf
from Metrics import utils

# Custom Matrix Multiplication uses one matrix w1 (m x n) as forward kernel and
# second matrix w2 (n x m) as backward kernel when calculating gradients
def custom_matmul(tie_grad=False):
    @tf.custom_gradient
    def matmul(x, w1, w2):

        # angle metrics
        def metrics(x, w1, w2, dx, dy):
            utils.tensor_angle(w1, tf.transpose(w2), 'weight_angle')
            utils.tensor_angle(tf.matmul(x, w1),
                               tf.matmul(x, tf.transpose(w2)),
                               'projection_angle')
            # Note this is not the angle between the true gradient and used
            # gradient, rather the angle between the local gradient calculauted
            # by backprop and the alignment strategy
            utils.tensor_angle(dx,
                               tf.matmul(dy, w1, transpose_b=True),
                               'gradient_angle')

            # utils.principal_angle(w1, tf.transpose(w2), 'projection_space')
            # utils.principal_angle(tf.transpose(w1), w2, 'gradient_space')


        # custom gradient
        def grad(dy):
            dx = tf.matmul(dy, w2)
            dw1 = tf.matmul(x, dy, transpose_a=True)
            dw2 = tf.zeros_like(w2)
            if tie_grad:
                dw2 += tf.transpose(dw1)
            metrics(x, w1, w2, dx, dy)

            return dx, dw1, dw2

        # result and gradient
        return tf.matmul(x, w1), grad
    return matmul


# Custom 2D Matrix Convolution uses one matrix w1 (h x w x c x f) as forward
# filter and second matrix w2 (h x w x c x f) as backward filter when
# calculating gradients
def custom_conv2d(strides=[1,1,1,1], padding="VALID", tie_grad=False):
    @tf.custom_gradient
    def conv2d(x, w1, w2):

        # angle metrics
        def metrics(x, w1, w2, dx, dy):
            utils.tensor_angle(w1, w2, 'weight_angle')
            utils.tensor_angle(tf.nn.conv2d(x, w1, strides, padding),
                          tf.nn.conv2d(x, w2, strides, padding),
                          'projection_angle')
            # Note this is not the angle between the true gradient and used
            # gradient, rather the angle between the local gradient calculauted
            # by backprop and the alignment strategy
            utils.tensor_angle(dx,
                tf.nn.conv2d_transpose(dy, w1, tf.shape(x), strides, padding),
                'gradient_angle')
            # Note principal angle does not work for convolutional layers

        # custom gradient
        def grad(dy):
            dx = tf.nn.conv2d_transpose(dy, w2, tf.shape(x), strides, padding)
            dw1 = tf.nn.conv2d_backprop_filter(x, w1.shape, dy, strides, padding)
            dw2 = tf.zeros_like(w2)
            if tie_grad:
                dw2 += dw1
            metrics(x, w1, w2, dx, dy)

            return dx, dw1, dw2

        # result and gradient
        return tf.nn.conv2d(x, w1, strides, padding), grad
    return conv2d
