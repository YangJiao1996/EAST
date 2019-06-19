import tensorflow as tf
import os
import glob

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')

FLAGS = tf.app.flags.FLAGS

input_node_names = ['model_0/concat']
output_node_names = ['model_0/feature_fusion/F_score', 'model_0/feature_fusion/F_geometry']

def main(arg=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    meta_path = os.path.join(FLAGS.checkpoint_path, 'model.ckpt-1011.{}'.format('meta'))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names
        )
        # Save the frozen grpah
        with open(os.path.join(FLAGS.checkpoint_path, 'saved_model.pb'), 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        os.path.join(FLAGS.checkpoint_path, 'saved_model.pb'),
        input_node_names,
        output_node_names)
    # converter.optimization = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tf_light_quant_model = converter.convert()
    open("converted_model.tflite", "wb").write(tf_light_quant_model)

if __name__ == '__main__':
    tf.app.run()