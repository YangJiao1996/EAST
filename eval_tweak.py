import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import locality_aware_nms as nms_locality
import lanms
import time

import model
from icdar import restore_rectangle, generate_rbox, load_annoataion, check_and_validate_polys

tf.app.flags.DEFINE_bool('use_gpu', False, 'whether to use gpu')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_string('test_path', '', '')

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'bmp', 'BMP']
    for parent, dirnames, filenames in os.walk(FLAGS.test_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def show_pairs(image_output, image_gt, save_name):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image_output, cmap='gray')
    axes[1].imshow(image_gt, cmap='gray')
    plt.savefig(save_name)
    plt.close()

def show_single(image, save_name):
    plt.imshow(image, cmap='gray')
    plt.savefig(save_name)
    plt.close()

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    if len(im.shape) == 3:
        h, w, _ = im.shape
    else:
        h, w = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)



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
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        im_fn_list = get_images()
        for im_fn in im_fn_list:
            # im_fn = FLAGS.image_file

            print(f"Processing {os.path.basename(im_fn)}...")
            imread_start = time.time()
            im = cv2.imread(im_fn)
            imread_end = time.time()
            imread_time = (imread_end - imread_start) * 1000
            im_h, im_w, _ = im.shape
            
            pair_flag = False
            txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[-1], 'txt')
            if os.path.exists(txt_fn):
                pair_flag = True
                text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (im_h, im_w))
                score_map_gt, geo_map_gt, training_mask_gt = generate_rbox((im_h, im_w), text_polys, text_tags)

            im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=384)
            
            network_fw_start = time.time()
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
            network_fw_end = time.time()
            network_fw_time = (network_fw_end - network_fw_start) * 1000
            print(f"Time elapsed in imread(): {imread_time:.2f}ms; network forward time {network_fw_time:.2f}ms.")
            
            # boxes = detect(score_map=score, geo_map=geometry)
            if len(score.shape) == 4:
                score = score[0, :, :, 0]
                geometry = geometry[0, :, :, ]
            # filter the score map
            xy_text = np.argwhere(score > 0.75)
            # sort the text boxes via the y axis
            xy_text = xy_text[np.argsort(xy_text[:, 0])]
            score_filtered = score > 0.75
            geometry_filtered = geometry * score_filtered[:, :, np.newaxis]

            im_fn_base, _ = os.path.splitext(os.path.basename(im_fn))
            
            score_name = im_fn_base + "_score.png"
            score_path = os.path.join(FLAGS.output_dir, "scores")
            if not os.path.exists(score_path):
                os.makedirs(score_path)
            score_file = os.path.join(score_path, score_name)
            if pair_flag:
                show_pairs(score_map_gt, score_filtered, score_file)
            else:
                show_single(score_filtered, score_file)
            for idx in range(5):
                geo_name = im_fn_base + "_geo" + str(idx) + ".png"
                geo_path = os.path.join(FLAGS.output_dir, "geometries")
                if not os.path.exists(geo_path):
                    os.makedirs(geo_path)
                geo_file = os.path.join(geo_name, geo_path)
                if pair_flag:
                    show_pairs(geo_map_gt[:, :, idx], geometry_filtered[:, :, idx], geo_file)
                else:
                    show_single(geometry_filtered[:, :, idx], geo_file)

        



        


if __name__ == '__main__':
    tf.app.run()