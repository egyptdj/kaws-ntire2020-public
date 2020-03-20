import tensorflow as tf


def average_gradients(gvs, scope='AverageGradients'):
    with tf.name_scope(scope):
        average_gvs = []
        for gv in zip(*gvs):
            gradient = []
            for g, _ in gv:
                gradient.append(tf.expand_dims(g, axis=-1))
            average_gradient = tf.reduce_mean(tf.concat(gradient, axis=-1), axis=-1)
            variable = gv[0][1]
            average_gvs.append((average_gradient, variable))
        return average_gvs
