import os
from random import randint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-54-1.1064.hdf5")
model = load_model(model_path)

def get_label(img_path):
    img = image.load_img(img_path,
                         target_size=(224, 224))
    original_image = image.img_to_array(img)
    original_image = preprocess_input(original_image)
    original_image = np.expand_dims(original_image, axis=0)

    preds = model.predict(original_image)
    pred = preds[0]
    print(pred)
    label = np.argmax(pred)

    return label

img_path_1 = "/home/ahahadelyaly/winter-camp-pek/food-101/food-101/new_images/correct_original/apple_pie/235537.jpg"
img_path_2 = "/home/ahahadelyaly/winter-camp-pek/food-101/food-101/new_images/correct_with_noise/apple_pie/235537.jpg"

get_label(img_path_1)
get_label(img_path_2)