import tensorflow as tf
import icdar
import cv2
import os
from tqdm import tqdm
import numpy as np

tf.app.flags.DEFINE_string("geo_maps_save_path","", "Path to save geometries")
tf.app.flags.DEFINE_string("score_maps_save_path","", "Path to save scores")
tf.app.flags.DEFINE_string("training_masks_save_path","", "Path to save training_masks")
tf.app.flags.DEFINE_string("geo_maps_vis_path","", "Path to save geometries visuals")
tf.app.flags.DEFINE_string("score_maps_vis_path","", "Path to save scores visuals")
tf.app.flags.DEFINE_bool("vis", False, "Whether to visualize geometries and scores")

FLAGS = tf.app.flags.FLAGS


def pixelwise_maps_visualization(score_map, geo_maps, image_filename):
    image_filename_base, _ = os.path.splitext(os.path.basename(image_filename))
    score_filename = image_filename_base + '_score.png'
    score_path = os.path.join(FLAGS.score_maps_vis_path, score_filename)
    cv2.imwrite(score_path, score_map*255)
    for idx in range(5):
        if idx == 4:
            geo_filename = image_filename_base + "_orientation.png"
        else:
            geo_filename = image_filename_base + f"_geo{idx}.png"
        geo_path = os.path.join(FLAGS.geo_maps_vis_path, geo_filename)
        cv2.imwrite(geo_path, geo_maps[:, :, idx])

def pixelwise_maps_saver(score_map, geo_maps, training_mask, image_filename):
    image_filename_base, _ = os.path.splitext(os.path.basename(image_filename))

    score_filename = image_filename_base + "_score"
    geo_map_filename = image_filename_base + "_geo"
    training_mask_filename = image_filename_base + "_mask"
    
    score_path = os.path.join(FLAGS.score_maps_save_path, score_filename)
    geo_path = os.path.join(FLAGS.geo_maps_save_path, geo_map_filename)
    training_mask_path = os.path.join(FLAGS.training_masks_save_path, training_mask_filename)

    np.save(score_path, score_map)
    np.save(geo_path, geo_maps)
    np.save(training_mask_path, training_mask)


def data_reader(input_size=512, visuliazation=True):
    image_list = np.array(icdar.get_images(FLAGS.training_data_path))
    print(f"{image_list.shape[0]} images found in {FLAGS.training_data_path}.")
    
    image_indices = np.arange(0, image_list.shape[0])

    np.random.shuffle(image_indices)

    images_original = []
    image_filenames = []
    score_maps = []
    geo_maps = []
    training_masks = []

    for image_index in tqdm(image_indices):
        image_filename = image_list[image_index]
        image_original = cv2.imread(image_filename)
        height_original, width_original, _ = image_original.shape
        label_filename = image_filename.replace(os.path.basename(image_filename).split('.')[-1], 'txt')
        if not os.path.exists(label_filename):
            print(f"Cannot find label for {image_filename}.")
            continue

        resize_ratio_3_x = input_size/float(width_original)
        resize_ratio_3_y = input_size/float(height_original)

        text_polygons, text_tags = icdar.load_annoataion(label_filename)
        text_polygons, text_tags = icdar.check_and_validate_polys(text_polygons, text_tags, \
                                        (height_original, width_original))
        image_resized, text_resized = icdar.resize_with_label(image_original, text_polygons, \
                                        resize_ratio_3_x, resize_ratio_3_y)
        ## Skip the weird cropping stage of data augementation
        score_map, geo_map, training_mask = icdar.generate_rbox((input_size, input_size),
                                            text_resized, text_tags)
        if FLAGS.vis == True:
            pixelwise_maps_visualization(score_map, geo_map, image_filename)
        
        pixelwise_maps_saver(score_map, geo_map, training_mask, image_filename)
        
        
    

def main(argv=None):
    if not os.path.exists(FLAGS.geo_maps_save_path):
        os.makedirs(FLAGS.geo_maps_save_path)
    if not os.path.exists(FLAGS.score_maps_save_path):
        os.makedirs(FLAGS.score_maps_save_path)
    if not os.path.exists(FLAGS.training_masks_save_path):
        os.makedirs(FLAGS.training_masks_save_path)
    if not os.path.exists(FLAGS.geo_maps_vis_path):
        os.makedirs(FLAGS.geo_maps_vis_path)
    if not os.path.exists(FLAGS.score_maps_vis_path):
        os.makedirs(FLAGS.score_maps_vis_path)
    data_reader()

if __name__ == '__main__':
    tf.app.run()