import os
import csv
import time
import argparse


def parse():
    parser = argparse.ArgumentParser(description='KAWS-NTIRE2020-EXTREMESR-SOLUTION')
    parser.add_argument('-g', '--gpu_ids', type=str, default='0', help='ids of the gpu to use. specify without whitespace (e.g. 023)')
    parser.add_argument('-i', '--num_iters', type=int, default=1500000, help='number of iterations')
    parser.add_argument('-k', '--total_images_k', type=int, default=7000, help='number of total images to train, specified by unit 1,000')
    parser.add_argument('-b', '--minibatch_size', type=int, default=1, help='size of the minibatch')
    parser.add_argument('-p', '--patch_size', type=int, default=24, help='size of the patch')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed to set')
    parser.add_argument('-t', '--transition', type=int, default=1000000, help='images to show before transition to next LOD')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='initial learning rate of the wavelet pyramid generators')
    parser.add_argument('-n', '--exp_name', type=str, default='kaws-ntire2020', help='experiment name')
    parser.add_argument('-dt', '--target_dir', type=str, default='./result', help='directory to store or access files')
    parser.add_argument('-dd', '--dataset_dir', type=str, default='./data/tfrecords', help='directory to load tfrecord dataset')
    parser.add_argument('--full_train', action='store_true', help='train the model including the curated training dataset')
    parser.add_argument('--break_stage', type=int, default=-1, help='break the training phase after this stage')
    parser.add_argument('--skip_train', action='store_true', help='flag to skip the training phase and run only the test phase')
    parser.add_argument('--resume_train', type=str, choices=['iter', 'lod', 'early'], default=None, help='resume training from stopped checkpoint')
    parser.add_argument('--test_image', type=str, default=None, help='path to the test images (specify by file or directory name)')
    args = parser.parse_args()
    args.target_dir = os.path.join(args.target_dir, args.exp_name)
    args.num_gpus = len(args.gpu_ids)
    os.makedirs(args.target_dir, exist_ok=True)
    with open(os.path.join(args.target_dir, 'argv.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())
        writer.writerow(['start_time', time.time()])
        writer.writerow(['-----', '-----'])
    return args
