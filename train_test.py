# %%
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_string('geometry', 'RBOX', '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
# tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/Models/EAST/checkpoints/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_boolean('debug_flag', False, 'whether to show debug messages')

import model
import icdar

FLAGS = tf.app.flags.FLAGS

# %% 
def tower_loss(images, score_maps, geo_maps, training_masks, training_flag, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=training_flag)
    print_shape_flag = tf.math.logical_and(tf.math.logical_not(training_flag), FLAGS.debug_flag)
    
    # Print out some debug messages
    score_maps = tf.cond(print_shape_flag, \
                         lambda: tf.Print(score_maps, [tf.shape(score_maps)], "tf - Shape of score_maps is :", summarize=4),\
                         lambda: tf.identity(score_maps))
    f_score = tf.cond(print_shape_flag, \
                         lambda: tf.Print(f_score, [tf.shape(f_score)], "tf - Shape of f_score is :", summarize=4),\
                         lambda: tf.identity(f_score))
    geo_maps = tf.cond(print_shape_flag, \
                         lambda: tf.Print(geo_maps, [tf.shape(geo_maps)], "tf - Shape of geo_maps is :", summarize=4),\
                         lambda: tf.identity(geo_maps))
    f_geometry = tf.cond(print_shape_flag, \
                         lambda: tf.Print(f_geometry, [tf.shape(f_geometry)], "tf - Shape of f_geometry is :", summarize=4),\
                         lambda: tf.identity(f_geometry))
    training_masks = tf.cond(print_shape_flag, \
                         lambda: tf.Print(training_masks, [tf.shape(training_masks)], "tf - Shape of training_masks is :", summarize=4),\
                         lambda: tf.identity(training_masks))
    model_loss = model.loss(score_maps, f_score,
                            geo_maps, f_geometry,
                            training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None and training_flag is True:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss

#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
