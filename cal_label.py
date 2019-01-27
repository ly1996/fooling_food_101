import os
from random import randint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np

model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-54-1.1064.hdf5")
inception_model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-inception-64-1.0762.hdf5")
model = load_model(inception_model_path)

def get_label_inception(img_path):
    img = image.load_img(img_path,
                         target_size=(299, 299))
    original_image = image.img_to_array(img)
    original_image = preprocess_input(original_image)
    original_image = np.expand_dims(original_image, axis=0)

    preds = model.predict(original_image)
    pred = preds[0]
    label = np.argmax(pred)

    return label

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
    print(label,pred[label])

    return label

# img_path_1 = "/home/ahahadelyaly/winter-camp-pek/food-101/food-101/new_images/correct_original/apple_pie/235537.jpg"
# img_path_2 = "/home/ahahadelyaly/winter-camp-pek/food-101/food-101/new_images/correct_with_noise/apple_pie/235537.jpg"
#
# get_label(img_path_1)
# get_label(img_path_2)

#测试resnet50 和 inception v3在正常的验证码库上的精确率

final_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/final_data_set")
normal_path = os.path.expanduser('~/winter-camp-pek/tmp/food-101/images')

count_correct = 0
count = 0

for root,dirs,files in os.walk(final_dir):
    dirs.sort()
    print(dirs)

    for i in range(len(dirs)):
        dir = dirs[i]
        sub_dir = os.path.join(final_dir, dir)
        normal_sub_dir = os.path.join(normal_path, dir)

        print(sub_dir)
        print(normal_sub_dir)

        for file in os.listdir(sub_dir):
            count += 1
            predict_class = get_label_inception(os.path.join(normal_sub_dir, file))
            if predict_class == i:
                count_correct += 1

    break

print(count)
print(count_correct)