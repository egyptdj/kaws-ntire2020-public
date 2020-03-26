import tensorflow as tf
from layer import *
from model import *
from util.image.process import *
from util.image.vgg import *
from util.train.multigpu import *


class Graph(object):
    def __init__(self, dataset, model, num_gpus, seed=None, scope='Graph'):
        super(Graph, self).__init__()
        assert num_gpus >= 0
        self.default = tf.get_default_graph()
        self.scope = scope
        self.dataset = dataset
        self.model = model
        self.num_gpus = num_gpus
        if seed is not None: tf.random.set_random_seed(seed)
        if self.num_gpus: self.devices = ['/gpu:{}'.format(i) for i in range(num_gpus)]
        else: self.devices = ['/cpu:0']

    def reset(self):
        tf.reset_default_graph()

    def finalize(self):
        self.default.finalize()

    def build(self, minibatch_size, patch_size=None):
        with tf.device('/cpu:0'):
            with tf.name_scope('Configure'):
                self.get_next_train_lod1, self.get_next_validation_lod1 = self.dataset.configure(lods=[0,1], minibatch_size=self.num_gpus*minibatch_size*256, patch_size=patch_size, augmentation=True, scope='LOD1')
                self.get_next_train_lod2, self.get_next_validation_lod2 = self.dataset.configure(lods=[0,1,2], minibatch_size=self.num_gpus*minibatch_size*64, patch_size=patch_size, augmentation=True, scope='LOD2')
                self.get_next_train_lod3, self.get_next_validation_lod3 = self.dataset.configure(lods=[0,1,2,3], minibatch_size=self.num_gpus*minibatch_size*16, patch_size=patch_size, augmentation=True, scope='LOD3')
                self.get_next_train_lod4, self.get_next_validation_lod4 = self.dataset.configure(lods=[0,1,2,3,4], minibatch_size=self.num_gpus*minibatch_size, patch_size=patch_size, augmentation=True, scope='LOD4')

            with tf.name_scope('Placeholders'):
                self.x0 = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='InputLOD0')
                self.x1 = tf.placeholder(tf.float32, shape=[None, patch_size*2, patch_size*2, 3], name='InputLOD1')
                self.x2 = tf.placeholder(tf.float32, shape=[None, patch_size*4, patch_size*4, 3], name='InputLOD2')
                self.x3 = tf.placeholder(tf.float32, shape=[None, patch_size*8, patch_size*8, 3], name='InputLOD3')
                self.x4 = tf.placeholder(tf.float32, shape=[None, patch_size*16, patch_size*16, 3], name='InputLOD4')
                if self.num_gpus: x0, x1, x2, x3, x4 = tf.split(self.x0, self.num_gpus), tf.split(self.x1, self.num_gpus), tf.split(self.x2, self.num_gpus), tf.split(self.x3, self.num_gpus), tf.split(self.x4, self.num_gpus)
                else: x0, x1, x2, x3, x4 = [self.x0], [self.x1], [self.x2], [self.x3], [self.x4]
                x0, x1, x2, x3, x4 = [expand(x) for x in x0], [expand(x) for x in x1], [expand(x) for x in x2], [expand(x) for x in x3], [expand(x) for x in x4]

            with tf.name_scope('Hyperparameters'):
                instance_norm_scale = False
                self.global_step = tf.Variable(initial_value=0, trainable=False, name='GlobalStep', shape=[], dtype=tf.int64)
                self.transition = tf.Variable(initial_value=0, trainable=False, name='Transition', shape=[], dtype=tf.int64)
                self.trained_images_refine = tf.Variable(initial_value=0, trainable=False, name='TrainedImagesRefine', shape=[], dtype=tf.int64)
                self.initial_learning_rate = tf.Variable(initial_value=0.0, trainable=False, name='InitialLearningRate', shape=[])
                learning_rate1 = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.dataset.trained_images_per_lod[3], decay_steps=500000, decay_rate=0.5, staircase=True)
                learning_rate2 = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.dataset.trained_images_per_lod[3], decay_steps=500000, decay_rate=0.5, staircase=True)
                learning_rate3 = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.dataset.trained_images_per_lod[3], decay_steps=500000, decay_rate=0.5, staircase=True)
                learning_rate4 = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.dataset.trained_images_per_lod[3], decay_steps=1000000, decay_rate=0.5, staircase=True)
                learning_rate_ref = tf.train.exponential_decay(1e-4, global_step=self.trained_images_refine, decay_steps=1000000, decay_rate=0.5, staircase=True)

        for id, device in enumerate(self.devices):
            with tf.device(device), tf.name_scope('GPU{}-Models'.format(id)):
                if id==0: reuse=False
                else: reuse=True
                with tf.name_scope('WaveDec'):
                    a1, h1, v1, d1 = dwt2d(x1[id])
                    a2, h2, v2, d2 = dwt2d(x2[id])
                    a3, h3, v3, d3 = dwt2d(x3[id])
                    a4, h4, v4, d4 = dwt2d(x4[id])

                with tf.name_scope('PyramidLOD1'):
                    x0up = x0[id]
                    a1up = 2.0 * x0up
                    sh1up, sv1up, sul1up, sur1up = sobel_edges(shrink(x0up))
                    (h1up, v1up, d1up), f1 = self.model.build_unet(tf.concat([-sh1up/2.0, -sv1up/2.0, -sul1up/2.0, -sur1up/2.0, a1up], axis=-1), downscale=False, reuse=reuse, scope='GeneratorLOD1')
                    d1up /= 2.0
                    x1up = idwt2d((a1up, h1up, v1up, d1up), scope='UpLOD1')
                    y1 = tf.clip_by_value(x1up, -1.0, 1.0, name='ResolvedLOD1')
                    tf.add_to_collection('RESOLVED_IMAGES1', y1)

                with tf.name_scope('PyramidLOD2'):
                    a2up = 2.0 * x1up
                    sh2up, sv2up, sul2up, sur2up = sobel_edges(shrink(x1up))
                    (h2up, v2up, d2up), f2 = self.model.build_unet(tf.concat([-sh2up/2.0, -sv2up/2.0, -sul2up/2.0, -sur2up/2.0, a2up], axis=-1), downscale=False, reuse=reuse, scope='GeneratorLOD2')
                    d2up /= 2.0
                    x2up = idwt2d((a2up, h2up, v2up, d2up), scope='UpLOD2')
                    y2 = tf.clip_by_value(x2up, -1.0, 1.0, name='ResolvedLOD2')
                    tf.add_to_collection('RESOLVED_IMAGES2', y2)

                with tf.name_scope('PyramidLOD3'):
                    a3up = 2.0 * x2up
                    sh3up, sv3up, sul3up, sur3up = sobel_edges(shrink(x2up))
                    (h3up, v3up, d3up), f3 = self.model.build_unet(tf.concat([-sh3up/2.0, -sv3up/2.0, -sul3up/2.0, -sur3up/2.0, a3up], axis=-1), downscale=False, reuse=reuse, scope='GeneratorLOD3')
                    d3up /= 2.0
                    x3up = idwt2d((a3up, h3up, v3up, d3up), scope='UpLOD3')
                    y3 = tf.clip_by_value(x3up, -1.0, 1.0, name='ResolvedLOD3')
                    tf.add_to_collection('RESOLVED_IMAGES3', y3)

                with tf.name_scope('PyramidLOD4'):
                    a4up = 2.0 * x3up
                    sh4up, sv4up, sul4up, sur4up = sobel_edges(shrink(x3up))
                    (h4up, v4up, d4up), _ = self.model.build_unet(tf.concat([-sh4up/2.0, -sv4up/2.0, -sul4up/2.0, -sur4up/2.0, a4up], axis=-1), downscale=True, concat_features=[f1, f2, f3], reuse=reuse, scope='GeneratorLOD4')
                    d4up /= 2.0
                    x4up = idwt2d((a4up, h4up, v4up, d4up), scope='UpLOD4')
                    y4 = tf.clip_by_value(x4up, -1.0, 1.0, name='ResolvedLOD4')
                    tf.add_to_collection('RESOLVED_IMAGES4', y4)

                with tf.name_scope('RefineLOD4'):
                    x4ref = self.model.build_edsr(y4, factor=1, scope='RefineLOD4')
                    y4ref = tf.clip_by_value(x4ref, -1.0, 1.0, name='RefinedLOD4')
                    tf.add_to_collection('REFINED_IMAGES', y4ref)

                with tf.name_scope('StyleModel'):
                    vgg = Vgg19('./cache/imagenet-vgg-verydeep-19.mat')
                    s4 = [vgg.build(tf.image.resize(shrink(h4, min=-1.5, max=1.5), [224, 224], 'bilinear')), vgg.build(tf.image.resize(shrink(v4, min=-1.5, max=1.5), [224, 224], 'bilinear')), vgg.build(tf.image.resize(shrink(d4), [224,224], 'bilinear'))]
                    s4up = [vgg.build(tf.image.resize(shrink(h4up, min=-1.5, max=1.5), [224, 224], 'bilinear')), vgg.build(tf.image.resize(shrink(v4up, min=-1.5, max=1.5), [224, 224], 'bilinear')), vgg.build(tf.image.resize(shrink(d4up), [224,224], 'bilinear'))]
                    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

            with tf.device(device), tf.name_scope('GPU{}-Losses'.format(id)):
                with tf.name_scope('LOD1'):
                    with tf.name_scope('Generate'):
                        loss_g1 = 1.0 * tf.reduce_mean(tf.abs(tf.concat([h1up, v1up, d1up], axis=-1)-tf.concat([h1, v1, d1], axis=-1)))
                        tf.add_to_collection('LOSS_G1', loss_g1)

                with tf.name_scope('LOD2'):
                    with tf.name_scope('Generate'):
                        loss_g2 = 4.0 * tf.reduce_mean(tf.abs(tf.concat([h2up, v2up, d2up], axis=-1)-tf.concat([h2, v2, d2], axis=-1)))
                        tf.add_to_collection('LOSS_G2', loss_g2)

                with tf.name_scope('LOD3'):
                    with tf.name_scope('Generate'):
                        loss_g3 = 16.0 * tf.reduce_mean(tf.abs(tf.concat([h3up, v3up, d3up], axis=-1)-tf.concat([h3, v3, d3], axis=-1)))
                        tf.add_to_collection('LOSS_G3', loss_g3)

                with tf.name_scope('LOD4'):
                    with tf.name_scope('Generate'):
                        loss_g4 = tf.reduce_mean([tf.reduce_mean([tf.sqrt(tf.reduce_mean(tf.square(gram_matrix(target[layer])-gram_matrix(recon[layer])))) for layer in style_layers]) for target, recon in zip(s4, s4up)])
                        tf.add_to_collection('LOSS_G4', loss_g4)

                    with tf.name_scope('Refine'):
                        loss_r = tf.reduce_mean(tf.abs(x4ref-x4[id]))
                        tf.add_to_collection('LOSS_R', loss_r)

        with tf.device('/cpu:0'):
            with tf.name_scope('Metrics'):
                with tf.name_scope('LOD1'):
                    self.y1 = shrink(tf.concat(tf.get_collection('RESOLVED_IMAGES1'), axis=0))
                    self.psnr1 = tf.reduce_mean(tf.image.psnr(self.x1, self.y1, 1.0))
                    self.ssim1 = tf.reduce_mean(tf.image.ssim(self.x1, self.y1, 1.0))
                    self.loss_g1 = tf.reduce_mean(tf.get_collection('LOSS_G1'))
                with tf.name_scope('LOD2'):
                    self.y2 = shrink(tf.concat(tf.get_collection('RESOLVED_IMAGES2'), axis=0))
                    self.psnr2 = tf.reduce_mean(tf.image.psnr(self.x2, self.y2, 1.0))
                    self.ssim2 = tf.reduce_mean(tf.image.ssim(self.x2, self.y2, 1.0))
                    self.loss_g2 = tf.reduce_mean(tf.get_collection('LOSS_G2'))
                with tf.name_scope('LOD3'):
                    self.y3 = shrink(tf.concat(tf.get_collection('RESOLVED_IMAGES3'), axis=0))
                    self.psnr3 = tf.reduce_mean(tf.image.psnr(self.x3, self.y3, 1.0))
                    self.ssim3 = tf.reduce_mean(tf.image.ssim(self.x3, self.y3, 1.0))
                    self.loss_g3 = tf.reduce_mean(tf.get_collection('LOSS_G3'))
                with tf.name_scope('LOD4'):
                    self.y4 = shrink(tf.concat(tf.get_collection('RESOLVED_IMAGES4'), axis=0))
                    self.psnr4 = tf.reduce_mean(tf.image.psnr(self.x4, self.y4, 1.0))
                    self.ssim4 = tf.reduce_mean(tf.image.ssim(self.x4, self.y4, 1.0))
                    self.loss_g4 = tf.reduce_mean(tf.get_collection('LOSS_G4'))
                with tf.name_scope('LOD4REF'):
                    self.y4ref = shrink(tf.concat(tf.get_collection('REFINED_IMAGES'), axis=0))
                    self.psnr4ref = tf.reduce_mean(tf.image.psnr(self.x4, self.y4ref, 1.0))
                    self.ssim4ref = tf.reduce_mean(tf.image.ssim(self.x4, self.y4ref, 1.0))
                    self.loss_r = tf.reduce_mean(tf.get_collection('LOSS_R'))
                with tf.name_scope('Objective'):
                    objective = self.psnr4ref
                    objective_moving_average = tf.train.ExponentialMovingAverage(0.9, name='MovingAverage')
                    with tf.control_dependencies([objective_moving_average.apply([objective])]):
                        self.objective_metric = tf.identity(objective_moving_average.average(objective), name='Metric')

        with tf.device('/cpu:0'):
            with tf.name_scope('Summary'):
                _ = tf.summary.scalar(name='minibatch_size', tensor=self.dataset.minibatch_size, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                _ = tf.summary.scalar(name='lods', tensor=self.dataset.lods, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                _ = tf.summary.scalar(name='global_step', tensor=self.global_step, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                with tf.name_scope('LOD1'):
                    _ = tf.summary.scalar(name='learning_rate', tensor=learning_rate1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                    _ = tf.summary.histogram(name='H', values=h1up, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='V', values=v1up, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='D', values=d1up, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='X', values=self.y1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_resolved')
                    _ = tf.summary.histogram(name='X', values=self.x1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_target')
                    _ = tf.summary.histogram(name='H', values=h1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='V', values=v1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='D', values=d1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.scalar(name='generate', tensor=self.loss_g1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='loss')
                    _ = tf.summary.scalar(name='psnr', tensor=self.psnr1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.scalar(name='ssim', tensor=self.ssim1, collections=['LOD1_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.image(name='H', tensor=h1up, max_outputs=2, family='detail_generated', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v1up, max_outputs=2, family='detail_generated', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d1up, max_outputs=2, family='detail_generated', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.y1, max_outputs=2, family='image_resolved', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.x1, max_outputs=2, family='image_target', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='H', tensor=h1, max_outputs=2, family='detail_target', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v1, max_outputs=2, family='detail_target', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d1, max_outputs=2, family='detail_target', collections=['LOD1_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    self.scalar_summary_lod1 = tf.summary.merge(tf.get_collection('LOD1_SCALAR_SUMMARIES'))
                    self.image_summary_lod1 = tf.summary.merge(tf.get_collection('LOD1_IMAGE_SUMMARIES'))
                with tf.name_scope('LOD2'):
                    _ = tf.summary.scalar(name='learning_rate', tensor=learning_rate2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                    _ = tf.summary.histogram(name='H', values=h2up, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='V', values=v2up, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='D', values=d2up, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='X', values=self.y2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_resolved')
                    _ = tf.summary.histogram(name='X', values=self.x2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_target')
                    _ = tf.summary.histogram(name='H', values=h2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='V', values=v2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='D', values=d2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.scalar(name='generate', tensor=self.loss_g2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='loss')
                    _ = tf.summary.scalar(name='psnr', tensor=self.psnr2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.scalar(name='ssim', tensor=self.ssim2, collections=['LOD2_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.image(name='H', tensor=h2up, max_outputs=2, family='detail_generated', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v2up, max_outputs=2, family='detail_generated', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d2up, max_outputs=2, family='detail_generated', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.y2, max_outputs=2, family='image_resolved', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.x2, max_outputs=2, family='image_target', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='H', tensor=h2, max_outputs=2, family='detail_target', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v2, max_outputs=2, family='detail_target', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d2, max_outputs=2, family='detail_target', collections=['LOD2_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    self.scalar_summary_lod2 = tf.summary.merge(tf.get_collection('LOD1_SCALAR_SUMMARIES')+tf.get_collection('LOD2_SCALAR_SUMMARIES'))
                    self.image_summary_lod2 = tf.summary.merge(tf.get_collection('LOD1_IMAGE_SUMMARIES')+tf.get_collection('LOD2_IMAGE_SUMMARIES'))
                with tf.name_scope('LOD3'):
                    _ = tf.summary.scalar(name='learning_rate', tensor=learning_rate3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                    _ = tf.summary.histogram(name='H', values=h3up, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='V', values=v3up, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='D', values=d3up, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='X', values=self.y3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_resolved')
                    _ = tf.summary.histogram(name='X', values=self.x3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_target')
                    _ = tf.summary.histogram(name='H', values=h3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='V', values=v3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='D', values=d3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.scalar(name='generate', tensor=self.loss_g3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='loss')
                    _ = tf.summary.scalar(name='psnr', tensor=self.psnr3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.scalar(name='ssim', tensor=self.ssim3, collections=['LOD3_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.image(name='H', tensor=h3up, max_outputs=2, family='detail_generated', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v3up, max_outputs=2, family='detail_generated', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d3up, max_outputs=2, family='detail_generated', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.y3, max_outputs=2, family='image_resolved', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.x3, max_outputs=2, family='image_target', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='H', tensor=h3, max_outputs=2, family='detail_target', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v3, max_outputs=2, family='detail_target', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d3, max_outputs=2, family='detail_target', collections=['LOD3_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    self.scalar_summary_lod3 = tf.summary.merge(tf.get_collection('LOD1_SCALAR_SUMMARIES')+tf.get_collection('LOD2_SCALAR_SUMMARIES')+tf.get_collection('LOD3_SCALAR_SUMMARIES'))
                    self.image_summary_lod3 = tf.summary.merge(tf.get_collection('LOD1_IMAGE_SUMMARIES')+tf.get_collection('LOD2_IMAGE_SUMMARIES')+tf.get_collection('LOD3_IMAGE_SUMMARIES'))
                with tf.name_scope('LOD4'):
                    _ = tf.summary.scalar(name='learning_rate', tensor=learning_rate4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                    _ = tf.summary.histogram(name='H', values=h4up, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='V', values=v4up, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='D', values=d4up, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_generated')
                    _ = tf.summary.histogram(name='X', values=self.y4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_resolved')
                    _ = tf.summary.histogram(name='X', values=self.x4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='image_target')
                    _ = tf.summary.histogram(name='H', values=h4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='V', values=v4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.histogram(name='D', values=d4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='detail_target')
                    _ = tf.summary.scalar(name='generate', tensor=self.loss_g4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='loss')
                    _ = tf.summary.scalar(name='psnr', tensor=self.psnr4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.scalar(name='ssim', tensor=self.ssim4, collections=['LOD4_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.image(name='H', tensor=h4up, max_outputs=2, family='detail_generated', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v4up, max_outputs=2, family='detail_generated', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d4up, max_outputs=2, family='detail_generated', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.y4, max_outputs=2, family='image_resolved', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='X', tensor=self.x4, max_outputs=2, family='image_target', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='H', tensor=h4, max_outputs=2, family='detail_target', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='V', tensor=v4, max_outputs=2, family='detail_target', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    _ = tf.summary.image(name='D', tensor=d4, max_outputs=2, family='detail_target', collections=['LOD4_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    self.scalar_summary_lod4 = tf.summary.merge(tf.get_collection('LOD1_SCALAR_SUMMARIES')+tf.get_collection('LOD2_SCALAR_SUMMARIES')+tf.get_collection('LOD3_SCALAR_SUMMARIES')+tf.get_collection('LOD4_SCALAR_SUMMARIES'))
                    self.image_summary_lod4 = tf.summary.merge(tf.get_collection('LOD1_IMAGE_SUMMARIES')+tf.get_collection('LOD2_IMAGE_SUMMARIES')+tf.get_collection('LOD3_IMAGE_SUMMARIES')+tf.get_collection('LOD4_IMAGE_SUMMARIES'))
                with tf.name_scope('LOD4REF'):
                    _ = tf.summary.scalar(name='learning_rate', tensor=learning_rate_ref, collections=['LOD4REF_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='hyperparameters')
                    _ = tf.summary.scalar(name='refine', tensor=self.loss_r, collections=['LOD4REF_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='loss')
                    _ = tf.summary.scalar(name='psnr_refined', tensor=self.psnr4ref, collections=['LOD4REF_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.scalar(name='ssim_refined', tensor=self.ssim4ref, collections=['LOD4REF_SCALAR_SUMMARIES', tf.GraphKeys.SUMMARIES], family='metrics')
                    _ = tf.summary.image(name='X', tensor=self.y4ref, max_outputs=2, family='image_refined', collections=['LOD4REF_IMAGE_SUMMARIES', tf.GraphKeys.SUMMARIES])
                    self.scalar_summary_lod4ref = tf.summary.merge(tf.get_collection('LOD1_SCALAR_SUMMARIES')+tf.get_collection('LOD2_SCALAR_SUMMARIES')+tf.get_collection('LOD3_SCALAR_SUMMARIES')+tf.get_collection('LOD4_SCALAR_SUMMARIES')+tf.get_collection('LOD4REF_SCALAR_SUMMARIES'))
                    self.image_summary_lod4ref = tf.summary.merge(tf.get_collection('LOD1_IMAGE_SUMMARIES')+tf.get_collection('LOD2_IMAGE_SUMMARIES')+tf.get_collection('LOD3_IMAGE_SUMMARIES')+tf.get_collection('LOD4_IMAGE_SUMMARIES')+tf.get_collection('LOD4REF_IMAGE_SUMMARIES'))


        trainable_variables = tf.trainable_variables()
        g_trainable_variables_lod1 = [v for v in trainable_variables if ('LOD1' in v.name) and ('Generator' in v.name)]
        g_trainable_variables_lod2 = [v for v in trainable_variables if ('LOD2' in v.name) and ('Generator' in v.name)]
        g_trainable_variables_lod3 = [v for v in trainable_variables if ('LOD3' in v.name) and ('Generator' in v.name)]
        g_trainable_variables_lod4 = [v for v in trainable_variables if ('LOD4' in v.name) and ('Generator' in v.name)]
        r_trainable_variables_lod4 = [v for v in trainable_variables if ('LOD4' in v.name) and ('Refine' in v.name)]
        self.model_parameters = np.sum([np.prod(v.shape.as_list()) for v in trainable_variables])
        self.model_parameter_bytes = np.sum([np.prod(v.shape.as_list() + [v.dtype.size]) for v in trainable_variables])

        for id, device in enumerate(self.devices):
            with tf.device(device), tf.name_scope('GPU{}-Optimizers'.format(id)):
                with tf.name_scope('LOD1'):
                    if id==0:
                        g_optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1, name='GeneratorOptimizer')
                    tf.add_to_collection('GRADIENT_G1', g_optimizer1.compute_gradients(loss=tf.get_collection('LOSS_G1')[id], var_list=g_trainable_variables_lod1))
                with tf.name_scope('LOD2'):
                    if id==0:
                        g_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2, name='GeneratorOptimizer')
                    tf.add_to_collection('GRADIENT_G2', g_optimizer2.compute_gradients(loss=tf.get_collection('LOSS_G1')[id]+tf.get_collection('LOSS_G2')[id], var_list=g_trainable_variables_lod1+g_trainable_variables_lod2))
                with tf.name_scope('LOD3'):
                    if id==0:
                        g_optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate3, name='GeneratorOptimizer')
                    tf.add_to_collection('GRADIENT_G3', g_optimizer3.compute_gradients(loss=tf.get_collection('LOSS_G1')[id]+tf.get_collection('LOSS_G2')[id]+tf.get_collection('LOSS_G3')[id], var_list=g_trainable_variables_lod1+g_trainable_variables_lod2+g_trainable_variables_lod3))
                with tf.name_scope('LOD4'):
                    if id==0:
                        g_optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate4, name='GeneratorOptimizer')
                        r_optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate_ref, name='RefineOptimizer')
                    tf.add_to_collection('GRADIENT_G4', g_optimizer4.compute_gradients(loss=tf.get_collection('LOSS_G4')[id], var_list=g_trainable_variables_lod4))
                    tf.add_to_collection('GRADIENT_R4', r_optimizer4.compute_gradients(loss=tf.get_collection('LOSS_R')[id], var_list=r_trainable_variables_lod4))

        with tf.device(self.devices[0]):
            with tf.name_scope('AverageGradients'):
                with tf.name_scope('LOD1'):
                    g_gv_lod1 = average_gradients(tf.get_collection('GRADIENT_G1'), scope='GeneratorGradient')
                with tf.name_scope('LOD2'):
                    g_gv_lod2 = average_gradients(tf.get_collection('GRADIENT_G2'), scope='GeneratorGradient')
                with tf.name_scope('LOD3'):
                    g_gv_lod3 = average_gradients(tf.get_collection('GRADIENT_G3'), scope='GeneratorGradient')
                with tf.name_scope('LOD4'):
                    g_gv_lod4 = average_gradients(tf.get_collection('GRADIENT_G4'), scope='GeneratorGradient')
                    r_gv_lod4 = average_gradients(tf.get_collection('GRADIENT_R4'), scope='RefineGradient')

        with tf.device(self.devices[0]):
            with tf.name_scope('Optimize'):
                with tf.name_scope('LOD1'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        g_optimize_lod1 = g_optimizer1.apply_gradients(g_gv_lod1, global_step=self.global_step, name='OptimizeGLOD1')
                    self.optimize_lod1 = g_optimize_lod1
                with tf.name_scope('LOD2'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        g_optimize_lod2 = g_optimizer2.apply_gradients(g_gv_lod2, global_step=self.global_step, name='OptimizeGLOD2')
                    self.optimize_lod2 = g_optimize_lod2
                with tf.name_scope('LOD3'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        g_optimize_lod3 = g_optimizer3.apply_gradients(g_gv_lod3, global_step=self.global_step, name='OptimizeGLOD3')
                    self.optimize_lod3 = g_optimize_lod3
                with tf.name_scope('LOD4'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        g_optimize_lod4 = g_optimizer4.apply_gradients(g_gv_lod4, global_step=self.global_step, name='OptimizeGLOD4')
                        with tf.control_dependencies([self.trained_images_refine.assign_add(self.num_gpus*minibatch_size)]):
                            r_optimize_lod4 = r_optimizer4.apply_gradients(r_gv_lod4, global_step=self.global_step, name='OptimizeRLOD4')
                    self.optimize_lod4 = [g_optimize_lod3, g_optimize_lod4]
                    self.optimize_lod4ref = r_optimize_lod4

        with tf.device(self.devices[0]):
            with tf.name_scope('InferenceModel'):
                self.infer_x = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='InferenceInput')
                infer_x = expand(self.infer_x)
                with tf.name_scope('PyramidLOD1'):
                    infer_x0up = infer_x
                    infer_a1up = 2.0 * infer_x0up
                    infer_sh1up, infer_sv1up, infer_sul1up, infer_sur1up = sobel_edges(shrink(infer_x0up))
                    (infer_h1up, infer_v1up, infer_d1up), infer_f1 = self.model.build_unet(tf.concat([-infer_sh1up/2.0, -infer_sv1up/2.0, -infer_sul1up/2.0, -infer_sur1up/2.0, infer_a1up], axis=-1), downscale=False, scope='GeneratorLOD1', reuse=True)
                    infer_d1up /= 2.0
                    infer_x1up = idwt2d((infer_a1up, infer_h1up, infer_v1up, infer_d1up), scope='UpLOD1')

                with tf.name_scope('PyramidLOD2'):
                    infer_a2up = 2.0 * infer_x1up
                    infer_sh2up, infer_sv2up, infer_sul2up, infer_sur2up = sobel_edges(shrink(infer_x1up))
                    (infer_h2up, infer_v2up, infer_d2up), infer_f2 = self.model.build_unet(tf.concat([-infer_sh2up/2.0, -infer_sv2up/2.0, -infer_sul2up/2.0, -infer_sur2up/2.0, infer_a2up], axis=-1), downscale=False, scope='GeneratorLOD2', reuse=True)
                    infer_d2up /= 2.0
                    infer_x2up = idwt2d((infer_a2up, infer_h2up, infer_v2up, infer_d2up), scope='UpLOD2')

                with tf.name_scope('PyramidLOD3'):
                    infer_a3up = 2.0 * infer_x2up
                    infer_sh3up, infer_sv3up, infer_sul3up, infer_sur3up = sobel_edges(shrink(infer_x2up))
                    (infer_h3up, infer_v3up, infer_d3up), infer_f3 = self.model.build_unet(tf.concat([-infer_sh3up/2.0, -infer_sv3up/2.0, -infer_sul3up/2.0, -infer_sur3up/2.0, infer_a3up], axis=-1), downscale=False, scope='GeneratorLOD3', reuse=True)
                    infer_d3up /= 2.0
                    infer_x3up = idwt2d((infer_a3up, infer_h3up, infer_v3up, infer_d3up), scope='UpLOD3')

                with tf.name_scope('PyramidLOD4'):
                    infer_a4up = 2.0 * infer_x3up
                    infer_sh4up, infer_sv4up, infer_sul4up, infer_sur4up = sobel_edges(shrink(infer_x3up))
                    (infer_h4up, infer_v4up, infer_d4up), _ = self.model.build_unet(tf.concat([-infer_sh4up/2.0, -infer_sv4up/2.0, -infer_sul4up/2.0, -infer_sur4up/2.0, infer_a4up], axis=-1), downscale=True, concat_features=[infer_f1, infer_f2, infer_f3], scope='GeneratorLOD4', reuse=True)
                    infer_d4up /= 2.0
                    infer_x4up = idwt2d((infer_a4up, infer_h4up, infer_v4up, infer_d4up), scope='UpLOD4')
                    infer_y4 = tf.clip_by_value(infer_x4up, -1.0, 1.0, name='ResolvedLOD4')

                with tf.name_scope('RefineLOD4'):
                    infer_x4ref = self.model.build_edsr(infer_y4, factor=1, scope='RefineLOD4', reuse=True)
                    infer_y4ref = tf.clip_by_value(infer_x4ref, -1.0, 1.0, name='RefinedLOD4')
                    infer_resolved_image = shrink(infer_y4ref)

                self.infer_resolved_image = tf.identity(infer_resolved_image, name='InferenceOutput')

        with tf.device('/cpu:0'):
            with tf.name_scope('Savers'):
                self.iter_saver = tf.train.Saver(name='IterSaver', max_to_keep=1)
                self.lod_saver = tf.train.Saver(name='LODSaver', max_to_keep=5)
                self.early_saver = tf.train.Saver(name='EarlySaver', max_to_keep=1)
