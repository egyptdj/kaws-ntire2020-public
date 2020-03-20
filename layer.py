# Implementation of low-level layers are adopted from the official repository of the StyleGAN, NVIDIA (https://github.com/NVlabs/stylegan).
# The adopted functions are modified to fit the 'NHWC' data format.
import numpy as np
import tensorflow as tf


#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessarily efficient or even meaningful.

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[3]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NHWC')

#----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, padding='SAME', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert padding in ['SAME', 'VALID', 'REFLECT', 'SYMMETRIC']
    w = get_weight([kernel, kernel, x.shape[3].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    if padding=='REFLECT' or padding=='SYMMETRIC':
        x = tf.pad(x, [[0,0], [(kernel-1)//2, (kernel-1)//2], [(kernel-1)//2, (kernel-1)//2], [0,0]], padding)
        padding = 'VALID'

    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding=padding, data_format='NHWC')

#----------------------------------------------------------------------------
# Atrous convolutional layer.

def atrous_conv2d(x, fmaps, kernel, rate, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[3].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.atrous_conv2d(x, w, rate, padding='SAME')

#----------------------------------------------------------------------------
# Subpixel convolutional layer.

def subpixel_conv2d(x, factor, fmaps=3, kernel=3, scope='SubpixelConv'):
    with tf.variable_scope(scope):
        y = tf.nn.depth_to_space(apply_bias(conv2d(x, fmaps=factor*factor*fmaps, kernel=kernel)), factor)
    return y

#----------------------------------------------------------------------------
# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[1:3]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    w = get_weight([kernel, kernel, x.shape[3].value, fmaps], **kwargs)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,2,2,1], padding='SAME', data_format='NHWC')

def conv2d_downscale2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[1:3]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    w = get_weight([kernel, kernel, x.shape[3].value, fmaps], **kwargs)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,2,2,1], padding='SAME', data_format='NHWC')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, 1, 1, -1])

#----------------------------------------------------------------------------
# ReLU activation. Just the same as tf.nn.relu()
def relu(x):
    return tf.nn.relu(x, name='ReLU')

#----------------------------------------------------------------------------
# Sigmoid activation. Just the same as tf.nn.sigmoid()
def sigmoid(x):
    return tf.nn.sigmoid(x, name='Sigmoid')

#----------------------------------------------------------------------------
# Tanh activation. Just the same as tf.nn.sigmoid()
def tanh(x):
    return tf.nn.tanh(x, name='Tanh')

#----------------------------------------------------------------------------
# Leaky ReLU activation. More efficient than tf.nn.leaky_relu() and supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Instance normalization.

def instance_norm(x, scale=1.0, epsilon=1e-8):
    assert len(x.shape)==4 # NHWC
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[1,2], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + epsilon)
        x *= scale
        x = tf.cast(x, orig_dtype)
        return x

#----------------------------------------------------------------------------
# Style modulation.

def style_mod(x, y, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(y, fmaps=x.shape[-1]*2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
        return x * (style[:,0] + 1) + style[:,1]

#----------------------------------------------------------------------------
# Global average pooling

def global_avg_pool(x):
    assert len(x.shape) == 4
    with tf.variable_scope('GlobalAvgPool'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x = tf.reduce_mean(x, axis=[1,2])
        x = tf.cast(x, orig_dtype)
        return x

#----------------------------------------------------------------------------
# Gram matrix

def gram_matrix(x):
    # result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    # input_shape = tf.shape(x)
    # num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    # return result/(num_locations)

    _x = tf.reshape(x, shape=[-1, x.shape[3]])
    y = tf.matmul(tf.transpose(_x), _x)
    return y
