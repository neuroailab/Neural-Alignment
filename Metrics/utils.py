import tensorflow as tf
import math

#### Performance

# Calculates top k accuracy
def accuracy(logits, labels, k=1):
    return tf.nn.in_top_k(logits, labels, k)


#### Angles

# Converts radians into degrees
def rad2deg(theta):
    return tf.multiply(tf.constant(180./math.pi), theta)

# Calculates angle between two tensors (A, B)
# with the same shape after they are flattened to vectors
def tensor_angle(A, B, name):

    flat_a = tf.reshape(A, [-1])
    flat_b = tf.reshape(B, [-1])

    normalized_a = tf.nn.l2_normalize(flat_a)
    normalized_b = tf.nn.l2_normalize(flat_b)

    cos_similarity = tf.reduce_sum(tf.multiply(normalized_a, normalized_b))
    cos_similarity = tf.clip_by_value(cos_similarity,
                                      tf.constant(-1.0),
                                      tf.constant(1.0))

    rad = tf.acos(cos_similarity)
    deg = rad2deg(rad)

    tf.identity(deg, name)

# Calculates max principal angle between two 2D tensors (A, B)
# where it is assumed A has shape (n, m) and B has shape (n, k)
def principal_angle(A, B, name):

    QA = tf.qr(A)
    QB = tf.qr(B)

    QAQB = tf.matmul(QA, QB, transpose_a=True)

    s = tf.svd(QAQB, compute_uv=False)

    p_angles = tf.acos(s)
    rad = tf.reduce_max(p_angles)
    deg = rad2deg(rad)

    tf.identity(deg, name)

# Calculates a measure of how "hebbian" an update
# Can't we do this through tensor_angle with modified input?
# def hebb_angle(x, y, kernel):
#     batch_outerprod = tf.einsum('ai,aj->aij', x,y) #x_i*y_j
#     angle_fn = lambda k: tensor_angle(k, kernel)
#     tf.identity(tf.map_fn(angle_fn, batch_outerprod), name='hebb_angle')


#### Statistics

# Calculates norm of difference between two tensors
def difference(a, b, ord='euclidean'):
    return tf.norm(tf.subtract(a - b), ord)

# Calculates various moment statistics of a tensor
def stats(tensor, name='weight'):
    tf.identity(tf.square(tf.norm(tensor, ord=2)),
                     name='{}sq_norm'.format(name))
    # Ommiting variance, because of difficulty aggregating
    sq_mean, _ = tf.nn.moments(tf.square(tensor),
                    axes=list(range(len(tensor.get_shape().as_list()))),
                    name='{}sq_moments'.format(name))
    tf.identity(sq_mean, name='{}sq_mean'.format(name))
    #tf.identity(sq_var, name='{}sq_var'.format(name))
