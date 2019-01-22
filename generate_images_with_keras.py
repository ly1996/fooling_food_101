import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from tensorflow.keras.models import load_model
from keras.preprocessing import image

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

def gen_fooling_images(model,x_input):
    print("call gen_fooling_images")

def main(_):
    print("enter main")
    target_size = (224, 224)
    input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images/nachos")
    model_path = ""

    eps = FLAGS.max_epsilon
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    print("batch_shape",batch_shape)

    # model = load_model(model_path)
    model = 1

    x_input = []
    idx = 0

    for file in os.listdir(input_dir):
        # print(file)
        result = re.findall(r"(.*).jpg", file)
        number = result[0]
        img = Image.open(os.path.join(input_dir, file))
        if img.size != target_size:
            img = img.resize(target_size)
        x = image.img_to_array(img)

        if idx < FLAGS.batch_size:
            x_input.append(x)
            idx = idx + 1
        else:
            gen_fooling_images(model,x_input)
            print(x_input.shape)
            x_input = []
            idx = 0

if __name__ == '__main__':
    tf.app.run()