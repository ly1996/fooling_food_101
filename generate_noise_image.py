import os
import shutil
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/correct_original")
noise_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/correct_with_noise")

eps = 16.0
num_iteration = 10
momentum = 1.0
alpha = eps / num_iteration

model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-54-1.1064.hdf5")
model = load_model(model_path)

def gen_noise_for_sub_dir(sub_dir,noise_sub_dir,i):
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = i

    cost_function = -tf.log(model_output_layer[0, object_type_to_fake])
    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    x_input = []
    idx = 0
    files = []
    for file in os.listdir(sub_dir):
        # print(file)
        img = image.load_img(os.path.join(sub_dir, file),
                             target_size=(224, 224))
        x = image.img_to_array(img)

        if idx < 16:
            x_input.append(x)
            idx = idx + 1
            files.append(os.path.join(noise_sub_dir, file))
        else:
            original_image = np.array(x_input)
            original_image = preprocess_input(original_image)

            x_min = original_image - eps
            x_max = original_image + eps

            hacked_image = np.copy(original_image)

            print(hacked_image.shape)

            batch_shape = [16, 224, 224, 3]
            grad = np.zeros(shape=batch_shape)

            cost = 0.0

            for i in range(num_iteration):
                # print(i)
                cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

                if np.isnan(np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True)).any():
                    break
                noise = gradients / np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True)
                # noise = gradients / tf.reduce_mean(tf.abs(gradients), [1, 2, 3], keep_dims=True)
                noise = momentum * grad + noise
                hacked_image = hacked_image + alpha * np.sign(noise)

                grad = noise

                # Move the hacked image one step further towards fooling the model
                # hacked_image += gradients * learning_rate

                # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
                hacked_image = np.clip(hacked_image, x_min, x_max)

            # print(cost)

            for i in range(16):
                img = hacked_image[i]

                # original_image = np.expand_dims(img, axis=0)
                # preds = model.predict(original_image)
                # print (np.argmax(preds[0]))

                mean = [103.939, 116.779, 123.68]
                img[..., 0] += mean[0]
                img[..., 1] += mean[1]
                img[..., 2] += mean[2]
                img = np.clip(img, 0, 255)
                # print(img.shape)
                # Save the hacked image!

                img_norm = np.copy(img)
                img_norm[..., 0] = img[..., 2]
                img_norm[..., 2] = img[..., 0]

                im = Image.fromarray(img_norm.astype(np.uint8))
                im.save(files[i])

            x_input = []
            idx = 0
            files = []

def gen_noise_for_single_image(img_path,target_path,i):
    # Grab a reference to the first and last layer of the neural net
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = i

    img = image.load_img(img_path,
                         target_size=(224, 224))
    original_image = image.img_to_array(img)

    original_image = preprocess_input(original_image)

    original_image = np.expand_dims(original_image, axis=0)

    x_min = original_image - eps
    x_max = original_image + eps

    hacked_image = np.copy(original_image)

    batch_shape = [1, 224, 224, 3]
    grad = np.zeros(shape=batch_shape)

    cost_function = -tf.log(model_output_layer[0, object_type_to_fake])
    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    cost = 0.0

    for i in range(num_iteration):
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        if np.isnan(np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True)).any():
            break
        noise = gradients / np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True)
        # noise = gradients / tf.reduce_mean(tf.abs(gradients), [1, 2, 3], keep_dims=True)
        noise = momentum * grad + noise
        hacked_image = hacked_image + alpha * np.sign(noise)

        grad = noise

        # Move the hacked image one step further towards fooling the model
        # hacked_image += gradients * learning_rate

        # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
        hacked_image = np.clip(hacked_image, x_min, x_max)

    print("Model's predicted likelihood that the image is a toaster: {:.8}".format(cost))

    img = hacked_image[0]
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    img = np.clip(img, 0, 255)
    print(img.shape)
    # Save the hacked image!

    img_norm = np.copy(img)
    img_norm[..., 0] = img[..., 2]
    img_norm[..., 2] = img[..., 0]

    im = Image.fromarray(img_norm.astype(np.uint8))
    im.save(target_path)

for root,dirs,files in os.walk(input_dir):
    dirs.sort()
    print(dirs)
    for i in range(len(dirs)):
        #beet_salad:5
        #beignets:6
        #bibimbap:7
        #bread_pudding:8
        #breakfast_burrito:9
        #bruschetta:10
        #caesar_salad:11
        #cannoli:12
        #caprese_salad:13
        #carrot_cake:14
        #ceviche:15
        #cheese_plate:16
        #cheesecake:17
        #chicken_curry:18
        #chicken_quesadilla:19
        #chicken_wings:20
        if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            continue
        dir = dirs[i]

        sub_dir = os.path.join(input_dir, dir)
        noise_sub_dir = os.path.join(noise_dir, dir)

        print(sub_dir)
        print(noise_sub_dir)

        if not os.path.exists(noise_sub_dir):
            os.makedirs(noise_sub_dir)

        # gen_noise_for_sub_dir(sub_dir,noise_sub_dir,i)

        idx = 0
        for file in os.listdir(sub_dir):
            idx += 1
            gen_noise_for_single_image(os.path.join(sub_dir, file),os.path.join(noise_sub_dir, file),i)
            if idx > 25:
                break
    break