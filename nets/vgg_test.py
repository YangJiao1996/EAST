import tensorflow as tf
import vgg
slim = tf.contrib.slim

is_training = False

with slim.arg_scope(vgg.vgg_arg_scope()):
    inputs = tf.placeholder(dtype = tf.float32, shape = [None, 512, 512, 3])
    net, end_points = vgg.vgg_16(inputs, is_training=is_training, num_classes=1, spatial_squeeze=False)

    for k in sorted(end_points.keys()):
        print(k, end_points[k].shape)

    print(net.shape)