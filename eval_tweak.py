import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import locality_aware_nms as nms_locality
import matplotlib.patches as patches
import lanms
import time
import csv

import model
from icdar import restore_rectangle, generate_rbox, load_annoataion, check_and_validate_polys

tf.app.flags.DEFINE_bool('use_gpu', False, 'whether to use gpu')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_string('test_path', '', 'Path to test images')
tf.app.flags.DEFINE_integer('num_threads', 4, 'Number of CPU threads to use')
tf.app.flags.DEFINE_string('im_name', '', 'File name of the test image')

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

def resize_by_input_size(image, labels_polygon, input_size=512):
    original_height, original_width, _ = image.shape
    resized_height = input_size
    resized_width = input_size
    resized_ratio_width = resized_width / original_width
    resized_ratio_height = resized_height / original_height
    resized_image = cv2.resize(image, dsize=(resized_width, resized_height))
    labels_polygon[:, :, 0] *= resized_ratio_width
    labels_polygon[:, :, 1] *= resized_ratio_height
    return image, labels_polygon, resized_height, resized_width
    

def detect(score_map, geo_map, score_map_thresh=0.75, box_thresh=0.01, nms_thres=0.1):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
        geo_map[:, :, 0:4] /= 2
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, which is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes

def show_pairs(image_output, image_gt, save_name=""):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image_output, cmap='gray')
    axes[1].imshow(image_gt, cmap='gray')
    plt.show()
    # plt.savefig(save_name)
    # plt.close()

def show_all(score_gt, score_rz, score_out, geo_map_gt, geo_map_rz, geo_map, save_name=""):
    fig, axes = plt.subplots(6, 3)
    axes[0][0].imshow(score_gt, cmap='gray')
    axes[0][1].imshow(score_rz, cmap='gray')
    axes[0][2].imshow(score_out, cmap='gray')
    for idx in range(1, 6):
        axes[idx][0].imshow(geo_map_gt[:, :, idx-1], cmap='gray')
        axes[idx][1].imshow(geo_map_rz[:, :, idx-1], cmap='gray')
        axes[idx][2].imshow(geo_map[:, :, idx-1], cmap='gray')
    plt.show()

def show_threes(image_output, image_resize, image_gt, save_name=""):
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image_output, cmap='gray')
    axes[2].imshow(image_gt, cmap='gray')
    axes[1].imshow(image_resize, cmap='gray')
    plt.show()

def show_single(image, save_name=""):
    plt.imshow(image, cmap='gray')
    plt.show()
    # plt.savefig(save_name)
    # plt.close()

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

def save_resize_stats(img_gt, img_gt_resized, img_in_resized, geo_map_gt, geo_map_rz, geo_map):
    img_gt_height, img_gt_width, _ = img_gt.shape
    img_rz_height, img_rz_width, _ = img_gt_resized.shape
    img_in_height, img_in_width, _ = img_in_resized.shape
    bbox_gt_height = geo_map_gt[:, :, 0].max()
    bbox_gt_width = geo_map_gt[:, :, 1].max()
    bbox_rz_height = geo_map_rz[:, :, 0].max()
    bbox_rz_width = geo_map_rz[:, :, 1].max()
    bbox_out_height = geo_map[:, :, 0].max()
    bbox_out_width = geo_map[:, :, 1].max()
    print(f" D[0]-Input-Org: {bbox_gt_height},  D[1]-Input-Org: {bbox_gt_width}")
    print(f"D[0]-Resize-512: {bbox_rz_height}, D[1]-Resize-512: {bbox_rz_width}")
    print(f"D[0]-Output-Ogn: {bbox_out_height}, D[1]-Output-Ogn: {bbox_out_width}")

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
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=FLAGS.num_threads)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
            # im_fn = FLAGS.image_file
        im_fn = os.path.join(FLAGS.test_path, FLAGS.im_name)
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

            im_gt_resized, resized_polylabels, resized_height, resized_width = resize_by_input_size(im, text_polys)
            score_map_rz, geo_map_rz, training_mask_rz = generate_rbox((resized_height, resized_width),\
                resized_polylabels, text_tags)

        im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=384)
        
        network_fw_start = time.time()
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        network_fw_end = time.time()
        network_fw_time = (network_fw_end - network_fw_start) * 1000
        print(f"Time elapsed in imread(): {imread_time:.2f}ms; network forward time {network_fw_time:.2f}ms.")



        boxes = detect(score_map=score, geo_map=geometry)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(im_resized, cmap='gray')
        for box in boxes:
            polygon = [box[0:2], box[2:4], box[4:6], box[6:8]]
            poly = patches.Polygon(polygon, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(poly)
            
        plt.show()
        if len(score.shape) == 4:
            score = score[0, :, :, 0]
            geometry = geometry[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score > 0.75)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        score_filtered = score > 0.75
        geometry_filtered = geometry * score_filtered[:, :, np.newaxis]
        save_resize_stats(im, im_gt_resized, im_resized, geo_map_gt, geo_map_rz, geometry_filtered)
        im_fn_base, _ = os.path.splitext(os.path.basename(im_fn))
        show_pairs(im, im_resized)
        score_name = im_fn_base + "_score.png"
        score_path = os.path.join(FLAGS.output_dir, "scores")
        if not os.path.exists(score_path):
            os.makedirs(score_path)
        score_file = os.path.join(score_path, score_name)
        if pair_flag:
            show_threes(score_map_gt, score_map_rz, score_filtered, score_file)
        else:
            show_single(score_filtered, score_file)
        for idx in range(5):
            geo_name = im_fn_base + "_geo" + str(idx) + ".png"
            geo_path = os.path.join(FLAGS.output_dir, "geometries")
            if not os.path.exists(geo_path):
                os.makedirs(geo_path)
            geo_file = os.path.join(geo_path, geo_name)
            if pair_flag:
                show_threes(geo_map_gt[:, :, idx], geo_map_rz[:, :, idx], geometry_filtered[:, :, idx], geo_file)
            else:
                show_single(geometry_filtered[:, :, idx], geo_file)

        



        


if __name__ == '__main__':
    tf.app.run()