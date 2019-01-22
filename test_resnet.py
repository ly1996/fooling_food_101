import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import backend as K

eps = 16.0
num_iteration = 10
momentum = 1.0
alpha = eps / num_iteration

model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-12-1.7504.hdf5")
model = load_model(model_path)

# Grab a reference to the first and last layer of the neural net
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

object_type_to_fake = 0
# Load the image to hack
img = image.load_img(os.path.expanduser("~/winter-camp-pek/food-101/food-101/images/apple_pie/1005649.jpg"), target_size=(224, 224))
original_image = image.img_to_array(img)
print(np.mean(original_image,axis=2))
print(np.mean(preprocess_input(original_image),axis=2))

# original_image = preprocess_input(original_image)

# Add a 4th dimension for batch size (as Keras expects)
original_image = np.expand_dims(original_image, axis=0)

x_min = img = np.clip(original_image - eps, 0, 255)
x_max = np.clip(original_image + eps, 0, 255)

# Create a copy of the input image to hack on
hacked_image = np.copy(original_image)

batch_shape = [1, 224 , 224, 3]
grad = np.zeros(shape=batch_shape)

print ("output shape",model_output_layer.shape)

# Define the cost function.
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
cost_function = -tf.log(model_output_layer[0, object_type_to_fake])
# np.log()

# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
gradient_function = K.gradients(cost_function, model_input_layer)[0]

# Create a Keras function that we can call to calculate the current cost and gradient
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

cost = 0.0

for i in range(num_iteration):
    print (i)

    # Check how close the image is to our target class and grab the gradients we
    # can use to push it one more step in that direction.
    # Note: It's really important to pass in '0' for the Keras learning mode here!
    # Keras layers behave differently in prediction vs. train modes!
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

    print(np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True))
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
# mean = [103.939, 116.779, 123.68]
# img[..., 0] += mean[0]
# img[..., 1] += mean[1]
# img[..., 2] += mean[2]
#
# img = np.clip(img, 0, 255)
# Save the hacked image!
im = Image.fromarray(img.astype(np.uint8))
im.save("hacked-image.jpg")