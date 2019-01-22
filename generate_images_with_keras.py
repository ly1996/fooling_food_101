import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import backend as K

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

def cal_one_hot(label,num_classes):
    # label = np.array([0, 3, 2, 8, 9, 1])  ##标签数据，标签从0开始
    classes = num_classes  ##类别数为最大数加1
    one_hot_label = np.zeros(shape=(label.shape[0], classes))  ##生成全0矩阵
    one_hot_label[np.arange(0, label.shape[0]), label] = 1  ##相应标签位置置1
    return one_hot_label

def cal_loss(preds, y):
    # los = []
    # len = preds.shape[0]
    # return -np.sum(y * np.log(preds), axis=1)
    # return np.mean(-np.sum(y * np.log(preds),axis=1))
    return tf.reduce_mean(-tf.reduce_sum(y * tf.log(preds), reduction_indices=[1]))

def gen_fooling_images(model,x_input,grad):
    # Grab a reference to the first and last layer of the neural net
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    eps = 16.0
    num_iteration = 10
    momentum = 1.0
    alpha = eps / num_iteration

    x_input = preprocess_input(x_input)
    x_min = x_input - eps
    x_max = x_input + eps

    print("call gen_fooling_images")

    #每张图片得到101维的属于各个类的概率
    preds = model.predict(x_input)

    #y :真实的类标
    y = np.argmax(preds,1)
    print (y)
    one_hot = cal_one_hot(y, 101)
    print(one_hot.shape)

    # cross_entropy = cal_loss(preds,one_hot)
    cross_entropy = cal_loss(model_output_layer,one_hot)
    # print (cross_entropy.shape)
    gradient_function = K.gradients(cross_entropy, model_input_layer)[0]
    grab_cost_and_gradients_from_mode = K.function([model_input_layer,K.learning_phase()], [cross_entropy,gradient_function])

    for i in range(num_iteration):
        # cost, noise = np.array(K.gradients(cross_entropy, x_input)[0])
        cost,noise = grab_cost_and_gradients_from_mode([x_input,0])
        print(noise.shape)

        noise = noise / np.mean(np.abs(noise), keep_dims=True)
        noise = momentum * grad + noise

        x_input = x_input + alpha * np.sign(noise)
        x_input = np.clip(x_input, x_min, x_max)

        # preds = model.predict(x_input)
        # print(np.argmax(preds,1))
        # cross_entropy = cal_loss(preds, one_hot)

    mean = [103.939, 116.779, 123.68]
    x_input[..., 0] += mean[0]
    x_input[..., 1] += mean[1]
    x_input[..., 2] += mean[2]

    x_input = tf.clip_by_value(x_input, 0, 255)

    return x_input

def save_img(new_image,file):
    output_dir = "~/winter-camp-pek/tmp"
    im = Image.fromarray(new_image)
    im.save(os.path.join(output_dir, file))

def main(_):
    print("enter main")
    target_size = (224, 224)
    input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images/nachos")
    model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-12-1.7504.hdf5")

    eps = FLAGS.max_epsilon
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    print("batch_shape",batch_shape)

    # model = 1
    model = load_model(model_path)

    x_input = []
    files = []
    idx = 0

    for file in os.listdir(input_dir):
        # print(file)
        # result = re.findall(r"(.*).jpg", file)
        # number = result[0]
        img = Image.open(os.path.join(input_dir, file))
        if img.size != target_size:
            img = img.resize(target_size)
        x = image.img_to_array(img)

        if idx < FLAGS.batch_size:
            x_input.append(x)
            files.append(file)
            idx = idx + 1
        else:
            grad = tf.zeros(shape=batch_shape)
            x_input = np.array(x_input)
            new_images = gen_fooling_images(model,x_input,grad)
            for i in range(FLAGS.batch_size):
                save_img(new_images[i], files[i])
            print(x_input.shape)
            x_input = []
            files = []
            idx = 0

if __name__ == '__main__':
    tf.app.run()