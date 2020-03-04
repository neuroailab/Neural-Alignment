from tensorflow.python.keras import activations
import tensorflow as tf
import primitives as reg
import numpy as np

# Alignment Base Class
class AlignmentBase(object):
    def __init__(self,
                 update_forward=False,
                 input_distribution=None,
                 input_stddev=1.0,
                 use_bias_forward=False,
                 use_bias_backward=False,
                 activation_fn_override=None,
                 activation_forward=False,
                 activation_backward=False,
                 batch_center_backward_input=False,
                 center_input=False,
                 normalize_input=False,
                 batch_center_forward_output=False,
                 center_forward_output=False,
                 normalize_forward_output=False,
                 center_backward_output=False,
                 normalize_backward_output=False):
        self.__name__ = type(self).__name__
        self.update_forward = update_forward
        self.input_distribution = input_distribution
        self.input_stddev = input_stddev
        self.use_bias_forward = use_bias_forward
        self.use_bias_backward = use_bias_backward
        self.activation_fn_override = activation_fn_override
        self.activation_forward = activation_forward
        self.activation_backward = activation_backward
        self.batch_center_backward_input = batch_center_backward_input
        self.center_input = center_input
        self.normalize_input = normalize_input
        self.batch_center_forward_output = batch_center_forward_output
        self.center_forward_output = center_forward_output
        self.normalize_forward_output = normalize_forward_output
        self.center_backward_output = center_backward_output
        self.normalize_backward_output = normalize_backward_output
        self.tie_grad = False

    def _batch_center(self, x):
        return x - tf.math.reduce_mean(x, axis=0, keepdims=True)

    def _center(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], -1])
        x = x - tf.math.reduce_mean(x, axis=1, keepdims=True)
        return tf.reshape(x, shape)

    def _normalize(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], -1])
        mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
        x = tf.math.l2_normalize(x - mean, axis=1)
        return tf.reshape(x, shape)

    def __call__(self,
                 input,
                 forward_kernel,
                 backward_kernel,
                 activation,
                 layer_type,
                 bias=None,
                 **kwargs):

        # Add stop gradients
        input = tf.stop_gradient(input)
        if bias is not None:
            bias = tf.stop_gradient(bias)
        if not self.update_forward:
            forward_kernel = tf.stop_gradient(forward_kernel)

        # Use random noise
        if self.input_distribution is not None:
            if self.input_distribution == 'uniform':
                val = np.sqrt(12 * self.input_stddev**2) / 2
                input = tf.random.uniform(tf.shape(input),
                                          minval=-val,
                                          maxval=val)
            elif self.input_distribution == 'normal':
                input = tf.random.normal(tf.shape(input),
                                         mean=0.0,
                                         stddev=self.input_stddev)
            else:
                raise ValueError

        # Standardize Input
        if self.center_input:
            input = self._center(input)
        if self.normalize_input:
            input = self._normalize(input)

        forward_input = tf.identity(input, name='forward_input')
        backward_input = tf.identity(input, name='backward_input')
        if self.batch_center_backward_input:
            backward_input = self._batch_center(backward_input)

        # Compute Projections
        if layer_type == 'fc':
            forward_projection = tf.matmul(forward_input, forward_kernel)
            backward_projection = tf.matmul(backward_input, backward_kernel,
                                            transpose_b=True)
            if bias is not None:
                if self.use_bias_forward:
                    forward_projection = tf.nn.bias_add(forward_projection, bias)
                if self.use_bias_backward:
                    backward_projection = tf.nn.bias_add(backward_projection, bias)
        elif layer_type == 'conv':
            forward_projection = tf.nn.conv2d(forward_input,
                                              forward_kernel,
                                              **kwargs)
            backward_projection = tf.nn.conv2d(backward_input,
                                               backward_kernel,
                                               **kwargs)
            if bias is not None:
                if self.use_bias_forward:
                    forward_projection = tf.nn.bias_add(forward_projection,
                                                        bias,
                                                        data_format='NHWC')
                if self.use_bias_backward:
                    backward_projection = tf.nn.bias_add(backward_projection,
                                                         bias,
                                                         data_format='NHWC')
        else:
            raise ValueError

        # Apply Activation
        if self.activation_fn_override is not None:
            activation = activations.get(self.activation_fn_override)
            print("Overriding the default activation function with {}".format(activation))
        if self.activation_forward and (activation is not None):
            forward_projection = activation(forward_projection)
            print("Using activation forward: {}".format(activation))
        if self.activation_backward and (activation is not None):
            backward_projection = activation(backward_projection)
            print("Using activation backward: {}".format(activation))

        # Compute Reconstruction
        if layer_type == 'fc':
            forward_reconstruction = tf.matmul(forward_projection, backward_kernel)
            backward_reconstruction = tf.matmul(backward_projection, forward_kernel,
                                                transpose_b=True)
        elif layer_type == 'conv':
            forward_reconstruction = tf.nn.conv2d_transpose(forward_projection,
                                                            backward_kernel,
                                                            tf.shape(input),
                                                            **kwargs)
            backward_reconstruction = tf.nn.conv2d_transpose(backward_projection,
                                                             forward_kernel,
                                                             tf.shape(input),
                                                             **kwargs)
        else:
            raise ValueError

        # Standardize forward output
        if self.center_forward_output:
            forward_projection = self._center(forward_projection)
            forward_reconstruction = self._center(forward_reconstruction)
        if self.normalize_forward_output:
            forward_projection = self._normalize(forward_projection)
            forward_reconstruction = self._normalize(forward_reconstruction)
        if self.batch_center_forward_output:
            forward_projection = self._batch_center(forward_projection)
            forward_reconstruction = self._batch_center(forward_reconstruction)

        # Standardize backward output
        if self.center_backward_output:
            backward_projection = self._center(backward_projection)
            backward_reconstruction = self._center(backward_reconstruction)
        if self.normalize_backward_output:
            backward_projection = self._normalize(backward_projection)
            backward_reconstruction = self._normalize(backward_reconstruction)

        # Reshape backward kernel
        if layer_type == 'fc':
            backward_kernel = tf.transpose(backward_kernel)
        elif layer_type == 'conv':
            pass
        else:
            raise ValueError

        # Add alignment loss
        loss = self.regularization(input,
                                   forward_kernel,
                                   backward_kernel,
                                   forward_projection,
                                   backward_projection,
                                   forward_reconstruction,
                                   backward_reconstruction)
        tf.add_to_collection('ALIGNMENT_LOSSES', loss)

    @property
    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        raise NotImplementedError

# Feedback Alignment
class Feedback(AlignmentBase):
    def __init__(self, **kwargs):
        super(Feedback, self).__init__(**kwargs)

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        return tf.constant(0.0)

# Symmetric Alignment
class Symmetric(AlignmentBase):
    def __init__(self,
                 alpha,
                 beta,
                 **kwargs):
        super(Symmetric, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        decay_loss = reg.decay(forward_kernel, backward_kernel)
        self_loss = reg.self(forward_kernel, backward_kernel)
        return self.alpha()*decay_loss + self.beta()*self_loss

# Activation Alignment
class Activation(AlignmentBase):
    def __init__(self,
                 alpha,
                 beta,
                 **kwargs):
        super(Activation, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        sparse_loss = reg.sparse(forward_projection,
                                              backward_projection)
        amp_loss = reg.amp(forward_projection, backward_projection)
        return self.alpha()*sparse_loss + self.beta()*amp_loss

# Weight Mirror
class Mirror(AlignmentBase):
    def __init__(self,
                 alpha,
                 beta,
                 **kwargs):
        super(Mirror, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        decay_loss = reg.decay(forward_kernel, backward_kernel)
        amp_loss = reg.amp(forward_projection, backward_projection)
        return self.alpha()*decay_loss + self.beta()*amp_loss

# Information Alignment
class Information(AlignmentBase):
    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 reconstruction_reversal=False,
                 reconstruction_amp=False,
                 use_sparse=False,
                 **kwargs):
        super(Information, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.reconstruction_reversal = reconstruction_reversal
        self.reconstruction_amp = reconstruction_amp
        self.use_sparse = use_sparse

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        if self.reconstruction_reversal:
            if self.reconstruction_amp:
                amp_loss =  reg.amp(input, backward_reconstruction)
            else:
                amp_loss =  reg.amp(forward_projection, backward_projection)
            null_loss = reg.null(tf.zeros_like(forward_reconstruction),
                                 backward_reconstruction)
        else:
            if self.reconstruction_amp:
                amp_loss =  reg.amp(input, forward_reconstruction)
            else:
                amp_loss =  reg.amp(forward_projection, backward_projection)
            null_loss = reg.null(forward_reconstruction,
                                 tf.zeros_like(backward_reconstruction))
        if self.use_sparse:
            decay_loss = reg.sparse(forward_projection,
                                    backward_projection)
        else:
            decay_loss = reg.decay(forward_kernel, backward_kernel)
        return self.alpha()*amp_loss + self.beta()*null_loss + self.gamma()*decay_loss

# Kolen Pollack Alignment
class Kolen_Pollack(AlignmentBase):
    def __init__(self,
                 alpha,
                 **kwargs):
        super(Kolen_Pollack, self).__init__(**kwargs)

        self.alpha = alpha
        self.tie_grad = True

    def regularization(self,
                       input,
                       forward_kernel,
                       backward_kernel,
                       forward_projection,
                       backward_projection,
                       forward_reconstruction,
                       backward_reconstruction):
        decay_loss = reg.decay(forward_kernel, backward_kernel)
        return self.alpha()*decay_loss
