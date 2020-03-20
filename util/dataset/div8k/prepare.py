import os
import zipfile
import requests
import PIL.Image
import numpy as np
import tensorflow as tf
from pywt import swt2
from tqdm import tqdm


# define dataset urls
train_url = ['http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_001to200.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_201to400.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_401to600.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_601to800.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_801to1000.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_1001to1200.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_1201to1400.zip',
'http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/trainHR_1401to1500.zip']
validation_url = ['http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR/validationLR.zip']


def download_data(target_dir='data'):
    # check directory paths
    target_dir = os.path.normpath(target_dir)
    os.makedirs(os.path.join(target_dir,'zip'), exist_ok=True)

    # download data
    print('downloading data from source url')
    for url in tqdm(train_url+validation_url):
        fname = url.split('/')[-1]
        data = requests.get(url)
        with open(os.path.join(target_dir, 'zip', fname), 'wb') as f: f.write(data.content)
        del data


def unzip_data(source_dir='data', target_dir='data'):
    # check directory paths
    source_dir = os.path.normpath(source_dir)
    target_dir = os.path.normpath(target_dir)
    assert os.path.isdir(os.path.join(source_dir, 'zip'))
    os.makedirs(os.path.join(target_dir, 'raw', 'trainHR'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'raw', 'validationLR'), exist_ok=True)

    # unzip data
    print('unzipping data')
    for zip in tqdm(os.listdir(os.path.join(source_dir, 'zip'))):
        # get split
        if 'train' in zip: split = 'trainHR'
        elif 'validation' in zip: split = ''
        else: raise

        # unzip file
        with zipfile.ZipFile(zip, 'r') as zf:
            zf.extractall(os.path.join(target_dir, 'raw', split))


def compute_mean(source_dir='data', curated=True):
    # check directory paths
    source_dir = os.path.normpath(source_dir)
    source_path = os.path.join(source_dir, 'raw', 'trainHR')
    assert os.path.isdir(os.path.join(source_dir, 'raw', 'trainHR'))

    # define variables
    pixels = 0
    values = np.zeros([3])
    imgs = os.listdir(source_path)
    if curated: imgs = [img for img in imgs if (img.startswith('14') and not img.startswith('1400')) or img.startswith('15')]

    # gather statistics
    print('computing rgb mean')
    for img in tqdm(imgs):
        img_array = np.asarray(PIL.Image.open(os.path.join(source_path, img)))
        pixels += img_array.shape[0]*img_array.shape[1]
        values += np.sum(img_array, axis=(0,1))

    return values/pixels


def convert_to_tfrecords(source_dir='data', target_dir='data', curated_num=100, numpy=True):
    # check directory paths
    source_dir = os.path.normpath(source_dir)
    target_dir = os.path.normpath(target_dir)
    assert os.path.isdir(os.path.join(source_dir, 'raw', 'trainHR')) and os.path.isdir(os.path.join(source_dir, 'raw', 'validationLR'))
    train_imgs = os.listdir(os.path.join(source_dir, 'raw', 'trainHR'))
    train_imgs.sort()
    validation_imgs = os.listdir(os.path.join(source_dir, 'raw', 'validationLR'))
    validation_imgs.sort()
    traincurated_imgs = train_imgs[-curated_num:]
    train_imgs = train_imgs[:-curated_num]
    lods = ['LR', 'LOD1', 'LOD2', 'LOD3', 'HR']
    os.makedirs(os.path.join(target_dir, 'tfrecords'), exist_ok=True)

    # define objects for exporting tfrecords
    tfr_options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.NONE)

    # iterate through train images
    print('exporting train dataset tfrecords')
    for i, img in enumerate(train_imgs):
        print('[{:4d}/{:4d}] adding {}\r'.format(i, len(train_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        tfr_train_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-train-{}.tfrecords'.format(img_id)), tfr_options)
        img_array = [PIL.Image.open(os.path.join(source_dir, 'raw', 'train{}'.format(lod), img)) for lod in lods]
        if numpy: img_array = [np.asarray(i) for i in img_array]
        features = tf.train.Features(feature={
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_id.encode()])),
            'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[0].tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[0].shape)),
            'dataLOD1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[1].tobytes()])), 'shapeLOD1': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[1].shape)),
            'dataLOD2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[2].tobytes()])), 'shapeLOD2': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[2].shape)),
            'dataLOD3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[3].tobytes()])), 'shapeLOD3': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[3].shape)),
            'dataLOD4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[4].tobytes()])), 'shapeLOD4': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[4].shape))})
        img_example = tf.train.Example(features=features)
        tfr_train_writer.write(img_example.SerializeToString())
        tfr_train_writer.close()

    # iterate through curated train images
    print('exporting curated train dataset tfrecords')
    for i, img in enumerate(traincurated_imgs):
        print('[{:4d}/{:4d}] adding {}\r'.format(i, len(traincurated_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        tfr_traincurated_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-traincurated-{}.tfrecords'.format(img_id)), tfr_options)
        img_array = [PIL.Image.open(os.path.join(source_dir, 'raw', 'train{}'.format(lod), img)) for lod in lods]
        if numpy: img_array = [np.asarray(i) for i in img_array]
        features = tf.train.Features(feature={
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_id.encode()])),
            'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[0].tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[0].shape)),
            'dataLOD1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[1].tobytes()])), 'shapeLOD1': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[1].shape)),
            'dataLOD2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[2].tobytes()])), 'shapeLOD2': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[2].shape)),
            'dataLOD3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[3].tobytes()])), 'shapeLOD3': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[3].shape)),
            'dataLOD4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array[4].tobytes()])), 'shapeLOD4': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array[4].shape))})
        img_example = tf.train.Example(features=features)
        tfr_traincurated_writer.write(img_example.SerializeToString())
        tfr_traincurated_writer.close()

    # iterate through validation images
    print('exporting validation dataset tfrecords')
    for i, img in enumerate(validation_imgs):
        print('[{:4d}/{:4d}] adding {}\r'.format(i, len(validation_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        tfr_validation_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-validation-{}.tfrecords'.format(img_id)), tfr_options)
        img_array = PIL.Image.open(os.path.join(source_dir, 'raw', 'validationLR', img))
        if numpy: img_array = np.asarray(img_array)
        features = tf.train.Features(feature={
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.split('.')[0].encode()])),
            'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array.tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array.shape))})
        img_example = tf.train.Example(features=features)
        tfr_validation_writer.write(img_example.SerializeToString())
        tfr_validation_writer.close()


def patch_convert_to_tfrecords(source_dir='data', target_dir='data', patch_size=32, stride=16, curated_num=100):
    # check directory paths
    source_dir = os.path.normpath(source_dir)
    target_dir = os.path.normpath(target_dir)
    assert os.path.isdir(os.path.join(source_dir, 'raw', 'trainHR')) and os.path.isdir(os.path.join(source_dir, 'raw', 'validationLR'))
    train_imgs = os.listdir(os.path.join(source_dir, 'raw', 'trainHR'))
    train_imgs.sort()
    validation_imgs = os.listdir(os.path.join(source_dir, 'raw', 'validationLR'))
    validation_imgs.sort()
    traincurated_imgs = train_imgs[-curated_num:]
    train_imgs = train_imgs[:-curated_num]
    lods = ['LR', 'LOD1', 'LOD2', 'LOD3', 'HR']
    os.makedirs(os.path.join(target_dir, 'tfrecords'), exist_ok=True)

    # define objects for exporting tfrecords
    tfr_options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.NONE)

    # iterate through train images
    print('exporting train dataset tfrecords')
    for i, img in enumerate(train_imgs):
        print('[{:4d}/{:4d}] adding {}'.format(i, len(train_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        img_array = [np.asarray(PIL.Image.open(os.path.join(source_dir, 'raw', 'train{}'.format(lod), img))) for lod in lods]
        img_height, img_width = img_array[0].shape[:2]
        height_offset_seq = np.append(np.arange(0, img_height-(patch_size+1), stride), img_height-patch_size)
        width_offset_seq = np.append(np.arange(0, img_width-(patch_size+1), stride), img_width-patch_size)
        patch_id = 0
        for height_offset in height_offset_seq:
            for width_offset in width_offset_seq:
                tfr_train_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-train-{}-P{}-{:04d}.tfrecords'.format(img_id, patch_size, patch_id)), tfr_options)
                patch_array = [x[height_offset*(2**l):(height_offset+patch_size)*(2**l), width_offset*(2**l):(width_offset+patch_size)*(2**l)] for l, x in enumerate(img_array)]
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_id.encode()])),
                    'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[0].tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[0].shape)),
                    'dataLOD1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[1].tobytes()])), 'shapeLOD1': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[1].shape)),
                    'dataLOD2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[2].tobytes()])), 'shapeLOD2': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[2].shape)),
                    'dataLOD3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[3].tobytes()])), 'shapeLOD3': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[3].shape)),
                    'dataLOD4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[4].tobytes()])), 'shapeLOD4': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[4].shape))})
                img_example = tf.train.Example(features=features)
                tfr_train_writer.write(img_example.SerializeToString())
                tfr_train_writer.close()
                patch_id += 1

    # iterate through curated train images
    print('exporting curated train dataset tfrecords')
    for i, img in enumerate(traincurated_imgs):
        print('[{:4d}/{:4d}] adding {}'.format(i, len(traincurated_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        img_array = [np.asarray(PIL.Image.open(os.path.join(source_dir, 'raw', 'train{}'.format(lod), img))) for lod in lods]
        img_height, img_width = img_array[0].shape[:2]
        height_offset_seq = np.append(np.arange(0, img_height-(patch_size+1), stride), img_height-patch_size)
        width_offset_seq = np.append(np.arange(0, img_width-(patch_size+1), stride), img_width-patch_size)
        patch_id = 0
        for height_offset in height_offset_seq:
            for width_offset in width_offset_seq:
                tfr_traincurated_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-traincurated-{}-P{}-{:04d}.tfrecords'.format(img_id, patch_size, patch_id)), tfr_options)
                patch_array = [x[height_offset*(2**l):(height_offset+patch_size)*(2**l), width_offset*(2**l):(width_offset+patch_size)*(2**l)] for l, x in enumerate(img_array)]
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_id.encode()])),
                    'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[0].tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[0].shape)),
                    'dataLOD1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[1].tobytes()])), 'shapeLOD1': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[1].shape)),
                    'dataLOD2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[2].tobytes()])), 'shapeLOD2': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[2].shape)),
                    'dataLOD3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[3].tobytes()])), 'shapeLOD3': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[3].shape)),
                    'dataLOD4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_array[4].tobytes()])), 'shapeLOD4': tf.train.Feature(int64_list=tf.train.Int64List(value=patch_array[4].shape))})
                img_example = tf.train.Example(features=features)
                tfr_traincurated_writer.write(img_example.SerializeToString())
                tfr_traincurated_writer.close()
                patch_id += 1

    # iterate through validation images
    print('exporting validation dataset tfrecords')
    for i, img in enumerate(validation_imgs):
        print('[{:4d}/{:4d}] adding {}'.format(i, len(validation_imgs), img), end='', flush=True)
        img_id = img.split('.')[0]
        tfr_validation_writer = tf.io.TFRecordWriter(os.path.join(target_dir, 'tfrecords', 'kaws-ntire2020-extreme-div8k-I-validation-{}.tfrecords'.format(img_id)), tfr_options)
        img_array = np.asarray(PIL.Image.open(os.path.join(source_dir, 'raw', 'validationLR', img)))
        features = tf.train.Features(feature={
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.split('.')[0].encode()])),
            'dataLOD0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array.tobytes()])), 'shapeLOD0': tf.train.Feature(int64_list=tf.train.Int64List(value=img_array.shape))})
        img_example = tf.train.Example(features=features)
        tfr_validation_writer.write(img_example.SerializeToString())
        tfr_validation_writer.close()
