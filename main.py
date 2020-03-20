import os
import csv
import random
import PIL.Image
import numpy as np
import tensorflow as tf
from dataset import *
from model import *
from graph import *
from session import *
from util.train import option
# from util.train import notifier


def main(notifier=None):
    # parse option
    args = option.parse()
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(args.gpu_ids)
    tf.logging.set_verbosity(tf.logging.ERROR)
    if notifier is not None: notifier.set_name(args.exp_name)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create paths
    for path_name in ['model', 'summary']: os.makedirs(os.path.join(args.target_dir, path_name), exist_ok=True)
    for path_name in ['iter', 'lod', 'early']: os.makedirs(os.path.join(args.target_dir, 'model', path_name), exist_ok=True)
    for path_name in ['train', 'validation', 'test']: os.makedirs(os.path.join(args.target_dir, 'summary', path_name), exist_ok=True)

    # define experiment
    dataset = DatasetDiv8k(source_dir=args.dataset_dir, full_train=args.full_train)
    model = Model()
    graph = Graph(dataset=dataset, model=model, num_gpus=args.num_gpus, seed=args.seed, scope='Graph')
    session = Session(notifier=notifier)

    # run train
    if not args.skip_train:
        graph.build(minibatch_size=args.minibatch_size, patch_size=args.patch_size)
        print('model parameters: {:,} count / {:,} bytes'.format(int(graph.model_parameters), graph.model_parameter_bytes))
        session.train(graph=graph, num_iters=args.num_iters, transition=args.transition, learning_rate=args.learning_rate, target_dir=args.target_dir, total_images_k=args.total_images_k, break_stage=args.break_stage, resume_train=args.resume_train)

    # run test
    if args.test_image is not None:
        graph.reset()
        session.reset(config=tf.ConfigProto(allow_soft_placement=True))
        session.restore_graph(target_dir=args.target_dir)

        if os.path.isfile(args.test_image): test_images = [args.test_image]
        elif os.path.isdir(args.test_image): test_images = [os.path.join(args.test_image, image_filename) for image_filename in sorted(os.listdir(args.test_image))]
        else: raise ValueError('File does not exist: {}'.format(args.test_image))

        timetable = {}
        for test_image in test_images:
            image_filename = test_image.split('/')[-1]

            print('resolving image {}'.format(image_filename))
            image = np.float32(PIL.Image.open(test_image))/255.0
            resolved_image, resolving_time = session.test(image=image, stride=12, weighting='cos', ensemble=True)
            final_image = PIL.Image.fromarray(resolved_image)
            final_image.save(os.path.join(args.target_dir, 'summary', 'test', image_filename), compress_level=0)

            print('finished in {:.4f} seconds'.format(resolving_time))
            timetable[image_filename] = resolving_time

        with open(os.path.join(args.target_dir, 'summary', 'test', 'timetable.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(timetable.items())

    # close session
    session.close()

if __name__=='__main__':
    main(notifier=None)
    exit(1)
