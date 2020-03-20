import os
import numpy as np
import tensorflow as tf


def parse_dataset(split, lods=None):
    def parse_train_tfrecords(record):
        raw_record = tf.io.parse_single_example(record, features={
            'id': tf.io.FixedLenFeature([], tf.string),
            'dataLOD0': tf.io.FixedLenFeature([], tf.string), 'shapeLOD0': tf.io.FixedLenFeature([3], tf.int64),
            'dataLOD1': tf.io.FixedLenFeature([], tf.string), 'shapeLOD1': tf.io.FixedLenFeature([3], tf.int64),
            'dataLOD2': tf.io.FixedLenFeature([], tf.string), 'shapeLOD2': tf.io.FixedLenFeature([3], tf.int64),
            'dataLOD3': tf.io.FixedLenFeature([], tf.string), 'shapeLOD3': tf.io.FixedLenFeature([3], tf.int64),
            'dataLOD4': tf.io.FixedLenFeature([], tf.string), 'shapeLOD4': tf.io.FixedLenFeature([3], tf.int64)})
        parsed_record = {}
        parsed_record['id'] = raw_record.get('id')
        parsed_record['shape'] = raw_record.get('shapeLOD0')
        for lod in lods:
            parsed_lod_image = tf.reshape(tf.decode_raw(raw_record.get('dataLOD{}'.format(lod)), tf.uint8), raw_record.get('shapeLOD{}'.format(lod)))
            parsed_lod_image.set_shape(raw_record.get('shapeLOD{}'))
            parsed_record['LOD{}'.format(lod)] = tf.image.convert_image_dtype(parsed_lod_image, dtype=tf.float32)
        return parsed_record

    def parse_test_tfrecords(record):
        raw_record = tf.io.parse_single_example(record, features={
            'id': tf.io.FixedLenFeature([], tf.string),
            'dataLOD0': tf.io.FixedLenFeature([], tf.string), 'shapeLOD0': tf.io.FixedLenFeature([3], tf.int64)})
        parsed_record = {}
        parsed_record['id'] = raw_record.get('id')
        parsed_record['shape'] = raw_record.get('shapeLOD0')
        parsed_image = tf.reshape(tf.decode_raw(raw_record.get('dataLOD0'), tf.uint8), raw_record.get('shapeLOD0'))
        parsed_image.set_shape(raw_record.get('shapeLOD'))
        parsed_record['LOD0'] = tf.image.convert_image_dtype(parsed_image, dtype=tf.float32)
        return parsed_record

    if split=='train' or split=='traincurated': return parse_train_tfrecords
    elif split=='validation' or split=='test': return parse_test_tfrecords
    else: raise


def create_patches(patch_size, lods):
    patch_size_tf = tf.constant(patch_size, dtype=tf.int32)
    def func(element):
        patch_dict = {'id': element['id']}
        image_height = tf.cast(element['shape'][0], tf.int32)
        image_width = tf.cast(element['shape'][1], tf.int32)
        height_offset = tf.random.uniform([], maxval=image_height-patch_size_tf, dtype=tf.int32)
        width_offset = tf.random.uniform([], maxval=image_width-patch_size_tf, dtype=tf.int32)
        for lod in lods:
            patch = tf.image.crop_to_bounding_box(element['LOD{}'.format(lod)], height_offset*(2**lod), width_offset*(2**lod), patch_size_tf*(2**lod), patch_size_tf*(2**lod))
            patch.set_shape([patch_size*(2**lod), patch_size*(2**lod), 3])
            patch_dict['LOD{}'.format(lod)] = patch
        return patch_dict
    return func


def augment_data(lods, flip=True, rotate=True):
    assert flip==True or rotate==True
    def func(element):
        k = tf.random.uniform([], maxval=8, dtype=tf.int32)
        for lod in lods:
            if flip and k//4==1:
                element['LOD{}'.format(lod)] = tf.image.flip_left_right(element['LOD{}'.format(lod)])
            if rotate:
                element['LOD{}'.format(lod)] = tf.image.rot90(element['LOD{}'.format(lod)], k%4)
        return element
    return func


class DatasetDiv8k(object):
    def __init__(self, source_dir='data/tfrecords', full_train=False, scope='Dataset'):
        super(DatasetDiv8k, self).__init__()
        with tf.name_scope(scope):
            self.lods = tf.Variable(0, trainable=False, name='LODs', dtype=tf.uint8)
            self.minibatch_size = tf.Variable(0, trainable=False, name='MinibatchSize', dtype=tf.uint16)
            self.trained_images = tf.Variable(0, trainable=False, name='TotalTrainedImages', dtype=tf.int64)
            self.trained_images_per_lod = [tf.Variable(0, trainable=False, name='TrainedImagesLOD1', dtype=tf.int64), tf.Variable(0, trainable=False, name='TrainedImagesLOD2', dtype=tf.int64), tf.Variable(0, trainable=False, name='TrainedImagesLOD3', dtype=tf.int64), tf.Variable(0, trainable=False, name='TrainedImagesLOD4', dtype=tf.int64)]
            if full_train: self.__train_dataset = tf.data.TFRecordDataset(tf.data.TFRecordDataset.list_files(os.path.join(os.path.normpath(source_dir), 'kaws-ntire2020-extreme-div8k-I-train*.tfrecords'), shuffle=True))
            else: self.__train_dataset = tf.data.TFRecordDataset(tf.data.TFRecordDataset.list_files(os.path.join(os.path.normpath(source_dir), 'kaws-ntire2020-extreme-div8k-I-train-*.tfrecords'), shuffle=True))
            self.__traincurated_dataset = tf.data.TFRecordDataset(tf.data.TFRecordDataset.list_files(os.path.join(os.path.normpath(source_dir), 'kaws-ntire2020-extreme-div8k-I-traincurated-*.tfrecords'), shuffle=True))

    def configure(self, lods, minibatch_size, patch_size=None, augmentation=False, scope='Configure'):
        with tf.name_scope(scope):
            with tf.name_scope('Parse'):
                self._train_dataset = self.__train_dataset.map(parse_dataset('train', lods), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                self._traincurated_dataset = self.__traincurated_dataset.map(parse_dataset('traincurated', lods), num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if patch_size:
                with tf.name_scope('Patch'):
                    self.patch_size = patch_size
                    self._train_dataset = self._train_dataset.map(create_patches(patch_size, lods), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    self._traincurated_dataset = self._traincurated_dataset.map(create_patches(patch_size, lods), num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if augmentation:
                with tf.name_scope('Augment'):
                    self._train_dataset = self._train_dataset.map(augment_data(lods))

            with tf.name_scope('Batch'):
                self._train_dataset = self._train_dataset.batch(minibatch_size, drop_remainder=True).repeat().prefetch(1)
                self._traincurated_dataset = self._traincurated_dataset.batch(minibatch_size, drop_remainder=True).repeat().prefetch(1)

            with tf.name_scope('MakeIterator'):
                self._train_iterator = self._train_dataset.make_one_shot_iterator()
                self._traincurated_iterator = self._traincurated_dataset.make_one_shot_iterator()

                # save attributes for direct access to get next ops
                with tf.control_dependencies([self.minibatch_size.assign(minibatch_size), self.lods.assign(max(lods)), self.trained_images.assign_add(minibatch_size)] + [_trained_images.assign_add(minibatch_size) for _trained_images in self.trained_images_per_lod[:max(lods)]]):
                    self.get_next_train = self._train_iterator.get_next()
                with tf.control_dependencies([self.minibatch_size.assign(minibatch_size), self.lods.assign(max(lods))]):
                    self.get_next_traincurated = self._traincurated_iterator.get_next()

        return self.get_next_train, self.get_next_traincurated
