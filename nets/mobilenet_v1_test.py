import tensorflow as tf
import mobilenet_v1

slim = tf.contrib.slim

is_training = False

with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)):
    inputs = tf.placeholder(dtype = tf.float32, shape = [None, 512, 512, 3])
    net, end_points = mobilenet_v1.mobilenet_v1_base(inputs)

    for k in sorted(end_points.keys()):
        print(k, end_points[k].shape)

    print(net.shape)