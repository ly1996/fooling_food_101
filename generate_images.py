import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from tensorflow.keras.models import load_model

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  print(input_dir)
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
    print(filepath)
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def graph(x, y, i, x_max, x_min, grad):
    return x,y,i,x_max,x_min,grad

def stop(x, y, i, x_max, x_min, grad):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)

def main(_):
    print("enter main")
    input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images/nachos")
    model_path = ""

    eps = FLAGS.max_epsilon
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        model = load_model(model_path)

        # with tf.Session() as sess:
        #     for filenames, images in load_images(input_dir, batch_shape):
        #         print ("shape", images.shape)
        #         adv_images = sess.run(x_adv, feed_dict={x_input: images})

if __name__ == '__main__':
    tf.app.run()
