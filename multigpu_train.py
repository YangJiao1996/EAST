import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import warnings

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_string('geometry', 'RBOX', '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/Models/EAST/checkpoints/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model', None, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_boolean('debug_flag', False, 'whether to show debug messages')

import model
import icdar

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(images, score_maps, geo_maps, training_masks, training_flag, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=training_flag)
    print_shape_flag = tf.math.logical_and(tf.math.logical_not(training_flag), FLAGS.debug_flag)
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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        # print(grad_and_vars)
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        # Only a work around solution
        if grads:
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MakeDirs(FLAGS.checkpoint_path)
    else:
        # TODO: probably not a good practice to delete all checkpoints in the directory
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    # training_flag = tf.placeholder(tf.bool, shape=(), name="training_flag")

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)


    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss = tower_loss(iis, isms, igms, itms, True, reuse_variables)
                print(f"Model loss: {model_loss}")
                print(f"Total loss: {total_loss}")
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                # total_loss_test, model_loss_test = tower_loss(iis, isms, igms, itms, False, reuse_variables)
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        pretrained_model = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(pretrained_model, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        training_data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        
        
        total_losses = []
        prev_losses_avg = 10e9
        start = time.time()
        for step in range(FLAGS.max_steps):
            print("Getting data...")
            training_data = next(training_data_generator)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: training_data[0],
                                                                                input_score_maps: training_data[2],
                                                                                input_geo_maps: training_data[3],
                                                                                input_training_masks: training_data[4]})
            total_losses.append(tl)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break
            if step % FLAGS.save_checkpoint_steps == 0:
                curr_losses_avg = np.average(total_losses)
                print(f"Average total loss of this Epoch: {curr_losses_avg:.4f}.", flush=True)
                if curr_losses_avg > prev_losses_avg:
                    print(f"Warning: total loss does not descend. No checkpoints are saved.")
                else:
                    print(f"Saving checkpoint to {FLAGS.checkpoint_path}")
                    saver.save(sess, FLAGS.checkpoint_path + f'model_{step}.ckpt', global_step=global_step)
                    prev_losses_avg = curr_losses_avg
                total_losses = []

            avg_time_per_step = (time.time() - start)
            avg_examples_per_second = (FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
            start = time.time()
            print('Step {:06d}: Model Loss {:.4f}, Total Loss {:.4f}. {:.2f} secs/step, {:.2f} examples/s'.format(
                step, ml, tl, avg_time_per_step, avg_examples_per_second), flush=True, end='\r')



            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: training_data[0],
                                                                                             input_score_maps: training_data[2],
                                                                                             input_geo_maps: training_data[3],
                                                                                             input_training_masks: training_data[4]})
                summary_writer.add_summary(summary_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()
