import numpy as np
import tensorflow as tf
from pywt import Wavelet


def expand(x, min=-1.0, max=1.0, scope='Expand'):
    assert x.shape.ndims==4
    assert max > min
    with tf.name_scope(scope):
        window = max-min
        center = (max+min)/2.0
        x = (x - (0.5 - center)) * window
        return x

def shrink(x, min=-1.0, max=1.0, scope='Shrink'):
    assert x.shape.ndims==4
    assert max > min
    with tf.name_scope(scope):
        window = max-min
        center = (max+min)/2.0
        x = (x / window) + (0.5 - center)
        return x


# soble_edges based on the original tensorflow implementation:
# https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/image_ops_impl.py#L3496-L3534
def sobel_edges(image):
    # Define vertical and horizontal Sobel filters.
    static_image_shape = image.get_shape()
    image_shape = tf.shape(image)
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
               [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = tf.constant(kernels, dtype=image.dtype)

    kernels_tf = tf.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
    padded = tf.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = tf.concat([image_shape, [num_kernels]], 0)
    output = tf.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return tf.unstack(output, axis=-1)


# dwt, idwt based on:
# https://github.com/wmylxmj/Discrete-Wavelet-Transform-2D/blob/master/wavelet.py
def dwt2d(x, wave='haar', scope='DWT2d'):
    w = Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)

    core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = tf.convert_to_tensor(kernel)
    with tf.variable_scope(scope):
        x3d = tf.expand_dims(x, 1)
        x3d = tf.transpose(x3d, [0,4,2,3,1])
        y3d = tf.nn.conv3d(x3d, kernel, padding='SAME', strides=[1, 1, 2, 2, 1])
        y3d = tf.transpose(y3d, [0,4,2,3,1])
        y = [tf.squeeze(y,1) for y in tf.split(y3d, y3d.shape[1], 1)]
    return y


def idwt2d(x, wave='haar', scope='InvDWT2d'):
    w = Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)

    core = np.zeros((ll.shape[0], ll.shape[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = tf.convert_to_tensor(kernel)

    with tf.variable_scope(scope):
        y = tf.stack([tf.stack([_x[...,c] for _x in x], axis=-1) for c in range(x[0].shape[3].value)], axis=1)
        output_shape = [tf.shape(y)[0], tf.shape(y)[1], 2*tf.shape(y)[2], 2*tf.shape(y)[3], 1]
        x3d = tf.nn.conv3d_transpose(y, kernel, output_shape=output_shape, padding='SAME', strides=[1,1,2,2,1])
        output = tf.transpose(tf.squeeze(x3d, -1), [0,2,3,1])
        n, h, w, c = x[0].get_shape()
        output.set_shape((n, 2*h, 2*w, c))
    return output
