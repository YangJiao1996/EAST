import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
import csv

import model
from icdar import restore_rectangle, generate_rbox, load_annoataion, check_and_validate_polys
import locality_aware_nms as nms_locality
import lanms


class EASTEval():
    def __init__(self, output_dirs, input_dirs):

        # Build the computational graph of the model
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.ground_truth_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='ground_truth_score_maps')
        self.ground_truth_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='ground_truth_geo_maps')
        self.ground_truth_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.output_score, self.output_geometry = model.model(input_images, is_training=False)
        self.variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)

        self.saver = tf.train.Saver(variable_averages.variables_to_restore())

        # Directories
        self.output_dirs = output_dirs
        check_output_dirs(output_dirs)
        self.input_dirs = input_dirs
        check_input_dirs(input_dirs)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.input_dirs["checkpoint"] = checkpoint_dir
    
    def set_test_dir(self, test_dir):
        self.input_dirs["test_images"] = test_dir
    
    def check_input_dirs(self, input_dirs):
        # TODO: Use exceptions instead of print messages to prompt users
        """ Check if directories for input data are valid. 
            Prompt the user to set a valid input directory.
        
        Arguments:
            input_dirs {[dict]} -- A dictionary storing the categories of input.
        """
        input_dirs_valid = True
        categories_to_check = ["checkpoint", "test_images"]
        for category_to_check in categories_to_check:
            if category_to_check in output_dirs:
                if not os.path.exists(input_dirs[category_to_check]):
                    print(f"Directory {input_dirs[category_to_check]} does not exist.")
                    input_dirs_valid = False
            else:
                print(f"Directory for {category_to_check} is not contained in your setting.")
                input_dirs_valid = False
        
        if not input_dirs_valid:
            print("Please provide a valid path to the directory before you proceed.")

    def check_output_dirs(self, output_dirs):
        """ Check and create directories to store the outcome.
        
        Arguments:
            output_categories {[dict]} -- A dictionary storing the categories of output.
        """
        # Check the directories (not a good practice)
        categories_to_check = ["scores", "boxes", "geometries"]

        for category_to_check in categories_to_check:
            if category_to_check in output_dirs:
                if not os.path.exists(output_dirs[category_to_check]):
                    os.makedirs(output_dirs[category_to_check])
                print(f"Directory {output_dirs[category_to_check]} exists or created.")

    def feed_network(self):
        """ Run the forward pass of the algorithm
        
        Returns:
            bounding_boxes {np.array} -- Reshaped bounding boxes
        """
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # Restore the model from checkpoint
            ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_dir)
            model_path = os.path.join(self.checkpoint_dir, os.path.basename(ckpt_state.model_checkpoint_path))

            self.saver.restore(sess, model_path)
            image_list = get_images()

            for image_fn in image_list:
                print(f"Processing {os.path.basename(image_fn)}...")

                # Load the image
                image_original = cv2.imread(image_fn)
                image_height, image_width, _ = image_original.shape
                # Load the ground truth (if exists)
                pair_flag = False # A flag showing if the ground truth exists
                label_fn = image_fn.replace(os.path.basename(image_fn).split('.')[-1], 'txt')

                if os.path.exists(label_fn):
                    pair_flag = True
                    label_polygons, label_tags = load_annoataion(label_fn)
                    label_polygons, label_tags = \
                        check_and_validate_polys(label_polygons, label_tags, (image_height, image_width))
                    score_map_gt, geo_map_gt, _ = \
                        generate_rbox((image_height, image_width), label_polygons, label_tags)
                
                image_resized, (resized_ratio_height, resized_ratio_width) = \
                    image_resize_fit(self, image_original, image_height, image_width, long_edge_limit=384)
                
                score_map, geo_map = sess.run([self.output_score, self.output_geometry], 
                                     feed_dict={input_images: [image_resized]})
                bounding_boxes = restore_bbox(score_map, geo_map)

                if bounding_boxes is not None:
                    bounding_boxes = bounding_boxes.reshape((-1, 4, 2))
                    bounding_boxes[:, :, 0] /= resized_ratio_width
                    bounding_boxes[:, :, 1] /= resized_ratio_height
                    return bounding_boxes
                else:
                    return None
                


    def get_images(self):
        """ Helper function to get all images in the test directory.
        
        Returns:
            [list] -- A list of paths to all the images
        """
        image_list = []
        image_exts = set(['jpg', 'png', 'jpeg', 'JPG', 'bmp', 'BMP'])
        for image_file in os.listdir(self.check_input_dirs["test_images"]): 
            _, image_ext = os.path.splitext(image_file)
            if image_ext in image_exts:
                image_list.append(image_file)

        print(f"{len(image_list)} images found in {self.check_input_dirs['test_images']}")
        return image_list

    
    def image_resize_fit(self, image, image_height, image_width, long_edge_limit=2400):
        """ Helper function to resize the image while fitting the size to the requirement of network architecture.
            The last feature map in the network is 32x smaller than the input image, which indicates that
            the length of both sizes of the input image should be a multiplier of 32.
        
        Arguments:
            image {np.array} -- The input image.
            image_height {int} -- The height of the input image.
            image_width {int} -- The width of the input image.
        
        Keyword Arguments:
            long_edge_limit {int} -- Limitation of the longer edge. 
                                     2400 means that the longer edge of the image should be shorter than 2400.
                                     (default: {2400})
        
        Returns:
            image_resized {np.array} -- The resized image.
            (resized_ratio_height, resized_ratio_width) {tuple} -- The resized ratio of both sides.
        """
        longer_edge = max(image_height, image_width)

        # Resize the image first
        if longer_edge > long_edge_limit:
            resize_ratio_overall = float(long_edge_limit) / longer_edge
        else:
            resize_ratio_overall = 1.

        # Using round() instead of int() here to keep the aspect ratio
        resized_height = round(image_height * resize_ratio_overall)
        resized_width = round(image_width * resize_ratio_overall)

        # Fit the length of sides into a multiplier of 32
        if resized_height % 32 != 0:
            resized_height = max(round(resized_height / 32) * 32, 32)
        if resized_width % 32 != 0:
            resized_width = max(round(resized_width / 32) * 32, 32)

        resized_ratio_height = resized_height / float(image_height)
        resized_ratio_width = resized_width / float(image_width)

        image_resized = cv2.resize(image, (int(resized_width), int(resized_height)))

        return image_resized, (resized_ratio_height, resized_ratio_width)

    def restore_bbox(self, score_map, geo_map, score_map_thresh=0.75, nms_thresh=0.1, box_region_thresh=0.1):
        """ Helper function to restore bounding boxes from the score map and geometry maps.
        
        Arguments:
            score_map {np.array} -- The score map generated by network. 4x smaller than original image in all edges.
            geo_map {np.array} -- Geometry maps generated by the network. 4x smaller than the original image in all edges.
        
        Keyword Arguments:
            score_map_thresh {float} -- Threshold for scores. (default: {0.75})
            nms_thresh {float} -- Threshold for NMS. (default: {0.1})
            box_region_thresh {float} -- Threshold for the average text scores in the box   (default: {0.1})
        
        Returns:
            bbox {np.array} -- Coordinates of four vertices of the bounding box
        """
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]

        # Filter the score map
        text_coords = np.argwhere(score_map  > score_map_thresh)
        text_coords = text_coords[np.argsort(text_coords[:, 0])]

        # Restore a bounding box for *each* pixel in the text region
        bbox_densed = restore_rectangle(text_coords[:, ::-1]*4, geo_map[text_coords[:, 0], text_coords[:, 1], :])
        print(f'{bbox_densed.shape[0]} text boxes restored before the NMS.')
        bboxes = np.zeros((bbox_densed.shape[0], 9), dtype=np.float32)
        bboxes[:, :8] = bbox_densed.reshape((-1, 8))
        bboxes[:, 8] = score_map[text_coords[:, 0], text_coords[:, 1]]

        bboxes = lanms.merge_quadrangle_n9(bboxes.astype('float32'), nms_thresh)

        if bboxes.shape[0] == 0:
            return None
        
        for i, bbox in enumerate(bboxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, bbox[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            bboxes[i, 8] = cv2.mean(score_map, mask)[0]
        bboxes = bboxes[bboxes[:, 8] > box_region_thresh]

        return bboxes