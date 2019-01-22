import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from tensorflow.keras.models import load_model

import re
from PIL import Image

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

def main(_):
    print("enter main")
    input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images/nachos")
    model_path = ""

    eps = FLAGS.max_epsilon
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    print("batch_shape",batch_shape)

    for file in os.listdir(input_dir):
        print(file)
        result = re.findall(r"(.*).jpg", file)
        number = result[0]
        img = Image.open(os.path.join(input_dir, file))
if __name__ == '__main__':
    tf.app.run()