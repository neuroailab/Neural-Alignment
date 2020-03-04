import tensorflow as tf

# ||W|| + ||B||
def decay(W, B):
    return tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(B))

# ||Wx|| + ||x.TB||
def sparse(Wx, xB):
    Wx = tf.reduce_sum(tf.reduce_mean(tf.square(Wx), 0))
    xB = tf.reduce_sum(tf.reduce_mean(tf.square(xB), 0))
    return Wx + xB

# ||BWx|| + ||x.TBW||
def null(BWx, xBW):
    BWx = tf.reduce_sum(tf.reduce_mean(tf.square(BWx), 0))
    xBW = tf.reduce_sum(tf.reduce_mean(tf.square(xBW), 0))
    return BWx + xBW

# -tr(BW)
def self(W, B):
    return -tf.reduce_sum(tf.multiply(W, B))

# -tr(x.TBWx)
def amp(Wx, xB):
    xBWx = tf.multiply(xB, Wx)
    return -tf.reduce_sum(tf.reduce_mean(xBWx, 0))
