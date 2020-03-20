import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.image.patch import *


class Session(object):
    def __init__(self, config=None, notifier=None):
        super(Session, self).__init__()
        self.config = config
        self.notifier = notifier
        self.sess = tf.Session(config=self.config)

    def __call__(self, config=None, notifier=None):
        self.sess.close()
        self.config = config
        self.notifier = notifier
        self.sess = tf.Session(config=self.config)

    def reset(self, config=None, notifier=None):
        self.__call__(config, notifier)

    def set_notifier(self, notifier):
        self.notifier = notifier

    def restore_graph(self, target_dir, mode=''):
        assert os.path.isdir(os.path.join(target_dir, 'model'))
        assert mode in ['', 'lod', 'iter', 'early']

        # restore model
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(target_dir, 'model', mode))
        meta_graph = tf.train.import_meta_graph('.'.join([latest_checkpoint, 'meta']))
        meta_graph.restore(sess=self.sess, save_path=latest_checkpoint)

    def train(self, graph, num_iters, transition, learning_rate, target_dir, total_images_k=None, break_stage=-1, resume_train=None):
        if resume_train:
            print('resuming train from saved {} checkpoint'.format(resume_train))
            assert resume_train in ['lod', 'iter', 'early']
            assert os.path.isdir(os.path.join(target_dir, 'model', resume_train))
            summary_writer_train = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'train'))
            summary_writer_validation = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'validation'))
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(target_dir, 'model', resume_train))
            graph.iter_saver.restore(sess=self.sess, save_path=latest_checkpoint)
        else:
            print('starting train from initial state')
            self.sess.run(tf.initializers.global_variables())
            self.sess.run(graph.transition.assign(transition))
            self.sess.run(graph.initial_learning_rate.assign(learning_rate))
            summary_writer_train = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'train'), graph=self.sess.graph)
            summary_writer_validation = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'validation'), graph=self.sess.graph)
            if self.notifier is not None:
                self.notifier.notify(msg='training session initialized\n=PARAMETERS=\n{:,} count\n{:,} bytes'.format(int(graph.model_parameters), graph.model_parameter_bytes))

        last_stage = 0
        largest_objective_metric = 0.0
        graph.finalize()
        transition = self.sess.run(graph.transition)
        current_iter = self.sess.run(graph.global_step)

        for iter in tqdm(range(current_iter, num_iters), ncols=50):
            trained_images = self.sess.run(graph.dataset.trained_images)
            if total_images_k is not None:
                if trained_images > 1000 * total_images_k:
                    print('breaking train loop since number of trained images exceeds total images')
                    break

            if trained_images < transition:
                train_image, validation_image = self.sess.run([graph.get_next_train_lod1, graph.get_next_validation_lod1])
                train_feed_dict = {
                    graph.x0: train_image['LOD0'],
                    graph.x1: train_image['LOD1']}
                validation_feed_dict = {
                    graph.x0: validation_image['LOD0'],
                    graph.x1: validation_image['LOD1']}
                scalar_summary, image_summary = graph.scalar_summary_lod1, graph.image_summary_lod1
                psnr, ssim = graph.psnr1, graph.ssim1
                optimize = graph.optimize_lod1
                current_stage = 1
            elif trained_images < 2*transition:
                train_image, validation_image = self.sess.run([graph.get_next_train_lod2, graph.get_next_validation_lod2])
                train_feed_dict = {
                    graph.x0: train_image['LOD0'],
                    graph.x1: train_image['LOD1'],
                    graph.x2: train_image['LOD2']}
                validation_feed_dict = {
                    graph.x0: validation_image['LOD0'],
                    graph.x1: validation_image['LOD1'],
                    graph.x2: validation_image['LOD2']}
                scalar_summary, image_summary = graph.scalar_summary_lod2, graph.image_summary_lod2
                psnr, ssim = graph.psnr2, graph.ssim2
                optimize = graph.optimize_lod2
                current_stage = 2
            elif trained_images < 3*transition:
                train_image, validation_image = self.sess.run([graph.get_next_train_lod3, graph.get_next_validation_lod3])
                train_feed_dict = {
                    graph.x0: train_image['LOD0'],
                    graph.x1: train_image['LOD1'],
                    graph.x2: train_image['LOD2'],
                    graph.x3: train_image['LOD3']}
                validation_feed_dict = {
                    graph.x0: validation_image['LOD0'],
                    graph.x1: validation_image['LOD1'],
                    graph.x2: validation_image['LOD2'],
                    graph.x3: validation_image['LOD3']}
                scalar_summary, image_summary = graph.scalar_summary_lod3, graph.image_summary_lod3
                psnr, ssim = graph.psnr3, graph.ssim3
                optimize = graph.optimize_lod3
                current_stage = 3
            else:
                train_image, validation_image = self.sess.run([graph.get_next_train_lod4, graph.get_next_validation_lod4])
                train_feed_dict = {
                    graph.x0: train_image['LOD0'],
                    graph.x1: train_image['LOD1'],
                    graph.x2: train_image['LOD2'],
                    graph.x3: train_image['LOD3'],
                    graph.x4: train_image['LOD4']}
                validation_feed_dict = {
                    graph.x0: validation_image['LOD0'],
                    graph.x1: validation_image['LOD1'],
                    graph.x2: validation_image['LOD2'],
                    graph.x3: validation_image['LOD3'],
                    graph.x4: validation_image['LOD4']}
                if trained_images < 5*transition:
                    scalar_summary, image_summary = graph.scalar_summary_lod4, graph.image_summary_lod4
                    psnr, ssim = graph.psnr4, graph.ssim4
                    optimize = graph.optimize_lod4
                    current_stage = 4
                else:
                    scalar_summary, image_summary = graph.scalar_summary_lod4ref, graph.image_summary_lod4ref
                    psnr, ssim = graph.psnr4ref, graph.ssim4ref
                    optimize = graph.optimize_lod4ref
                    current_stage = 5

            # functions called once at stage transition
            if current_stage > last_stage and iter > 0:
                graph.lod_saver.save(self.sess, os.path.join(os.path.normpath(target_dir), 'model', 'lod', 'model_lod{}.ckpt'.format(last_stage)))
                if last_stage == break_stage: break
                else:
                    summary_writer_train = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'train'), graph=self.sess.graph)
                    summary_writer_validation = tf.summary.FileWriter(os.path.join(os.path.normpath(target_dir), 'summary', 'validation'), graph=self.sess.graph)
            last_stage = current_stage

            # optimize and write summary
            train_scalar_summary, _ = self.sess.run([scalar_summary, optimize], feed_dict=train_feed_dict)
            summary_writer_train.add_summary(train_scalar_summary, trained_images)
            validation_scalar_summary = self.sess.run(scalar_summary, feed_dict=validation_feed_dict)
            summary_writer_validation.add_summary(validation_scalar_summary, trained_images)

            # early stop saver at the last stage
            if current_stage==5:
                current_objective_metric = self.sess.run(graph.objective_metric, feed_dict=validation_feed_dict)
                if current_objective_metric.mean()>largest_objective_metric:
                    largest_objective_metric = current_objective_metric
                    graph.early_saver.save(self.sess, os.path.join(os.path.normpath(target_dir), 'model', 'early', 'model_early.ckpt'))

            # write image summary and notify at every 1000 iters
            if iter%1000==0:
                train_image_summary = self.sess.run(image_summary, feed_dict=train_feed_dict)
                summary_writer_train.add_summary(train_image_summary, trained_images)
                validation_image_summary = self.sess.run(image_summary, feed_dict=validation_feed_dict)
                summary_writer_validation.add_summary(validation_image_summary, trained_images)
                graph.iter_saver.save(self.sess, os.path.join(os.path.normpath(target_dir), 'model', 'iter', 'model_iter.ckpt'), global_step=graph.dataset.trained_images)
                if self.notifier is not None:
                    t_psnr, t_ssim = self.sess.run([psnr, ssim], feed_dict=train_feed_dict)
                    v_psnr, v_ssim = self.sess.run([psnr, ssim], feed_dict=validation_feed_dict)
                    self.notifier.notify(msg='iter:{:,}k\nimages_trained:{:,}k\n=TRAIN=\npsnr:{:.4f}\nssim:{:.4f}\n=VALIDATION=\npsnr:{:.4f}\nssim:{:.4f}'.format(iter//1000, trained_images//1000, t_psnr, t_ssim, v_psnr, v_ssim))

        graph.iter_saver.save(self.sess, os.path.join(os.path.normpath(target_dir), 'model', 'model_final.ckpt'))
        summary_writer_train.close()
        summary_writer_validation.close()


    def test(self, image, stride=None, weighting=None, ensemble=False):
        # fetch inference nodes
        input = self.sess.graph.get_tensor_by_name('InferenceModel/InferenceInput:0')
        resolved_image = self.sess.graph.get_tensor_by_name('InferenceModel/InferenceOutput:0')
        patch_size = input.shape.as_list()[1]

        # resolve image
        if not ensemble:
            start_time = time.time()
            patch_processed_image = extract_patches(image, patch_size, stride)
            patch_resolved_image = []
            for patch in tqdm(patch_processed_image, ncols=50):
                patch_resolved_image.append(resolved_image.eval({input: patch[np.newaxis,...]}, session=self.sess))
            patch_resolved_image = np.concatenate(patch_resolved_image, axis=0)
            recon_image = reconstruct_patches(patch_resolved_image, image_size=(image.shape[0]*16, image.shape[1]*16), stride=stride*16, weighting=weighting)
            final_image = np.rint(recon_image*255.0).astype('uint8')
            end_time = time.time()
        else:
            start_time = time.time()
            patch_processed_image = extract_patches(image, patch_size, stride)
            ensemble_patch_processed_image = [
                patch_processed_image, \
                np.rot90(patch_processed_image, k=1, axes=(1,2)), \
                np.rot90(patch_processed_image, k=2, axes=(1,2)), \
                np.rot90(patch_processed_image, k=3, axes=(1,2)), \
                np.fliplr(patch_processed_image), \
                np.fliplr(np.rot90(patch_processed_image, k=1, axes=(1,2))), \
                np.fliplr(np.rot90(patch_processed_image, k=2, axes=(1,2))), \
                np.fliplr(np.rot90(patch_processed_image, k=3, axes=(1,2)))]

            ensemble_patch_resolved_image = []
            for patches in ensemble_patch_processed_image:
                patch_resolved_image = []
                for patch in tqdm(patches, ncols=50):
                    patch_resolved_image.append(resolved_image.eval({input: patch[np.newaxis,...]}, session=self.sess))
                patch_resolved_image = np.concatenate(patch_resolved_image, axis=0)
                ensemble_patch_resolved_image.append(patch_resolved_image)

            final_patch_resolved_image = [
                ensemble_patch_resolved_image[0], \
                np.rot90(ensemble_patch_resolved_image[1], k=-1, axes=(1,2)), \
                np.rot90(ensemble_patch_resolved_image[2], k=-2, axes=(1,2)), \
                np.rot90(ensemble_patch_resolved_image[3], k=-3, axes=(1,2)), \
                np.fliplr(ensemble_patch_resolved_image[4]), \
                np.rot90(np.fliplr(ensemble_patch_resolved_image[5]), k=-1, axes=(1,2)), \
                np.rot90(np.fliplr(ensemble_patch_resolved_image[6]), k=-2, axes=(1,2)), \
                np.rot90(np.fliplr(ensemble_patch_resolved_image[7]), k=-3, axes=(1,2))]

            final_image = np.rint(np.mean([reconstruct_patches(patches, image_size=(image.shape[0]*16, image.shape[1]*16), stride=stride*16, weighting=weighting) for patches in final_patch_resolved_image], axis=0)*255.0).astype(np.uint8)
            end_time = time.time()

        return final_image, end_time-start_time

    def close(self):
        self.sess.close()
