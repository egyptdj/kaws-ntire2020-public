import tensorflow as tf
from layer import *


def edsr_block(x, fmaps=256, kernel=3, scaling=0.1, scope='EDSRBlock'):
    assert len(x.shape)==4 and x.shape[3]==fmaps
    with tf.variable_scope(scope):
        _x = x
        with tf.variable_scope('Conv0'):
            _x = apply_bias(conv2d(_x, fmaps, kernel))
            _x = relu(_x)
        with tf.variable_scope('Conv1'):
            _x = apply_bias(conv2d(_x, fmaps, kernel))
        with tf.variable_scope('Scale'):
            _x *= scaling
        with tf.variable_scope('ResSum'):
            x += _x
        return x


def rcab_block(x, fmaps=64, kernel=3, reduction_ratio=16, scope='RCABBlock'):
    assert len(x.shape)==4 and x.shape[3]==fmaps
    with tf.variable_scope(scope):
        _x = x
        with tf.variable_scope('Conv0'):
            _x = apply_bias(conv2d(_x, fmaps, kernel))
            _x = relu(_x)
        with tf.variable_scope('Conv1'):
            _x = apply_bias(conv2d(_x, fmaps, kernel))
        with tf.variable_scope('ChAttn'):
            __x = global_avg_pool(_x)
            with tf.variable_scope('Conv2'):
                __x = apply_bias(conv2d(__x, fmaps//reduction_ratio, 1))
                __x = relu(__x)
            with tf.variable_scope('Conv3'):
                __x = apply_bias(conv2d(__x, fmaps, 1))
                __x = sigmoid(__x)
        with tf.variable_scope('Scale'):
            _x *= __x
        with tf.variable_scope('ResSum'):
            x += _x
        return x


def residual_dense_block(x, growth_rate=64, kernel=3, layers=8, scaling=1.0, scope='ResidualDenseBlock'):
    assert x.shape.ndims==4
    with tf.variable_scope(scope):
        _x = x
        for i in range(layers):
            with tf.variable_scope('Conv{}'.format(i)):
                __x = apply_bias(conv2d(_x, fmaps=growth_rate, kernel=kernel))
                __x = relu(__x)
                _x = tf.concat([_x, __x], axis=-1)
        with tf.variable_scope('LocalFeatureFusion'):
            _x = apply_bias(conv2d(_x, 64, 1))
            x += scaling * _x
        return x


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self.training = None

    def build_unet(self, x, fmaps=64, stages=4, activation='relu', downscale=True, concat_features=None, instance_normalization=False, scope='UNET', reuse=tf.AUTO_REUSE):
        assert activation in ['relu', 'lrelu']
        if activation=='relu': activation = relu
        if activation=='lrelu': activation = leaky_relu
        with tf.variable_scope(scope, reuse=reuse):
            _x = x
            features = []
            for stage in range(stages):
                with tf.variable_scope('Stage{}'.format(stage)):
                    with tf.variable_scope('Conv0'):
                        _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**stage), kernel=3, padding='REFLECT'))
                        _x = activation(_x)
                    with tf.variable_scope('Conv1'):
                        _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**stage), kernel=3, padding='REFLECT'))
                        _x = activation(_x)
                    features.append(_x)
                    if downscale: _x = downscale2d(_x)

            with tf.variable_scope('Stage{}'.format(stages)):
                with tf.variable_scope('Conv0'):
                    _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**stages), kernel=3, padding='REFLECT'))
                    if instance_normalization: _x = instance_norm(_x)
                    _x = activation(_x)
                with tf.variable_scope('Conv1'):
                    _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**(stages-1)), kernel=3, padding='REFLECT'))
                    if instance_normalization: _x = instance_norm(_x)
                    _x = activation(_x)

            for stage in range(stages-1, 0, -1):
                with tf.variable_scope('Stage{}Up'.format(stage)):
                    if concat_features is not None: _x = tf.concat([concat_features[stages-stage-1], features[stage], upscale2d(_x)], axis=-1)
                    else: _x = tf.concat([features[stage], _x], axis=-1)
                    with tf.variable_scope('Conv0'):
                        _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**stage), kernel=3, padding='REFLECT'))
                        _x = activation(_x)
                    with tf.variable_scope('Conv1'):
                        _x = apply_bias(conv2d(_x, fmaps=fmaps*(2**(stage-1)), kernel=3, padding='REFLECT'))
                        _x = activation(_x)

            with tf.variable_scope('Stage0Up'):
                if downscale: _x = tf.concat([features[0], upscale2d(_x)], axis=-1)
                else: _x = tf.concat([features[0], _x], axis=-1)
                with tf.variable_scope('Conv0'):
                    _x = apply_bias(conv2d(_x, fmaps=fmaps, kernel=3, padding='REFLECT'))
                    _x = activation(_x)
                with tf.variable_scope('Conv1'):
                    _x = apply_bias(conv2d(_x, fmaps=fmaps, kernel=3, padding='REFLECT'))
                    _x = activation(_x)
                    if concat_features is None: concat_features = _x

            with tf.variable_scope('FeatureFusion'):
                _x = apply_bias(conv2d(_x, fmaps=9, kernel=1))

            y = tf.split(_x, 3, axis=-1)
            return y, concat_features


    def build_edsr(self, x, factor, fmaps=256, num_blocks=32, scaling=0.1, scope='EDSR', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            _x = x
            with tf.variable_scope('Conv0'):
                _x = relu(apply_bias(conv2d(_x, fmaps=fmaps, kernel=1)))
            __x = _x
            for block in range(num_blocks):
                with tf.variable_scope('Block{}'.format(block)):
                    __x = edsr_block(__x, fmaps=fmaps, scaling=scaling)
            with tf.variable_scope('ResSum'):
                _x += __x
            if factor==1:
                y = apply_bias(conv2d(_x, fmaps=x.shape[-1].value, kernel=1))
            else:
                y = subpixel_conv2d(_x, factor, fmaps=x.shape[-1].value)
            return y
