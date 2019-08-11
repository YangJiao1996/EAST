import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import locality_aware_nms as nms_locality
import lanms
import time
import eval_utils

import model
from icdar import restore_rectangle, generate_rbox, load_annoataion, check_and_validate_polys

tf.app.flags.DEFINE_bool('use_gpu', False, 'whether to use gpu')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_string('test_path', '', '')
tf.app.flags.DEFINE_integer('num_threads', 4, 'Number of CPU threads to use')
tf.app.flags.DEFINE_float('iou_threshold', 0.75, "Threshold of IoU in average precision calculation")

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    if FLAGS.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    ground_truth_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='ground_truth_score_maps')
    ground_truth_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='ground_truth_geo_maps')
    ground_truth_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    geo_path = os.path.join(FLAGS.output_dir, "geometries")
    score_path = os.path.join(FLAGS.output_dir, "scores")
    res_path = os.path.join(FLAGS.output_dir, "boxes")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    if not os.path.exists(score_path):
        os.makedirs(score_path)
    if not os.path.exists(geo_path):
        os.makedirs(geo_path)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=FLAGS.num_threads)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        im_fn_list = eval_utils.get_images(FLAGS.test_path)
        for im_fn in im_fn_list:
            print(f"Processing {os.path.basename(im_fn)}...")
            imread_start = time.time()
            im = cv2.imread(im_fn)
            imread_end = time.time()
            imread_time = (imread_end - imread_start) * 1000
            im_h, im_w, _ = im.shape
            
            txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[-1], 'txt')
            if os.path.exists(txt_fn):
                text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (im_h, im_w))
                score_map_gt, geo_map_gt, training_mask_gt = generate_rbox((im_h, im_w), text_polys, text_tags)
            else:
                print(f"File {txt_fn} not found: unable to evalutate")
                continue

            im_resized, (ratio_h, ratio_w) = eval_utils.resize_image(im, max_side_len=384)
            
            network_fw_start = time.time()
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
            network_fw_end = time.time()
            network_fw_time = (network_fw_end - network_fw_start) * 1000
            
            
            post_proc_start = time.time()
            boxes = eval_utils.detect(score_map=score, geo_map=geometry)
            post_proc_end = time.time()
            post_proc_time = (post_proc_end - post_proc_start) * 1000
            print(f"Time elapsed in imread(): {imread_time:.2f}ms; network forward: {network_fw_time:.2f}ms; post-processing: {post_proc_time:.2f}ms.")
            if len(score.shape) == 4:
                score = score[0, :, :, 0]
                geometry = geometry[0, :, :, ]
            score_filtered = score > 0.75
            geometry_filtered = geometry * score_filtered[:, :, np.newaxis]

            im_fn_base, _ = os.path.splitext(os.path.basename(im_fn))
            
            score_name = im_fn_base + "_score.png"

            score_file = os.path.join(score_path, score_name)
            eval_utils.show_pairs(score_map_gt, score_filtered, score_file)

            for idx in range(5):
                geo_name = im_fn_base + "_geo" + str(idx) + ".png"
                geo_file = os.path.join(geo_path, geo_name)
                eval_utils.show_pairs(geo_map_gt[:, :, idx], geometry_filtered[:, :, idx], geo_file)

            average_precisions = []

            if boxes is not None:
                average_precision = eval_utils.average_precision_image(boxes, text_polys, text_tags, im_w, im_h, \
                                                                ratio_w, ratio_h, threshold=FLAGS.iou_threshold)
                print(f"Average precision of the image: {average_precision}.")
                average_precisions.append(average_precision)
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                res_name = im_fn_base + "_box.png"
                res_file = os.path.join(res_path, res_name)
                eval_utils.show_polygons(im, boxes, text_polys, text_tags, average_precision, res_file)

    mean_average_precision = np.mean(average_precisions)        
    print(f"Mean Average Precision: {mean_average_precision:.4f}")


if __name__ == '__main__':
    tf.app.run()