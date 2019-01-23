import os
import shutil
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images")
test_input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/test-set")
correct_output_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/correct_original")
incorrect_output_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/incorrect")
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
    label = np.argmax(pred)

    return label

for root,dirs,files in os.walk(input_dir):
    dirs.sort()
    print(dirs)
    mix_types = []
    for i in range(len(dirs)):
        dir = dirs[i]

        sub_dir = os.path.join(input_dir,dir)
        test_sub_dir = os.path.join(test_input_dir,dir)
        correct_target_sub_dir = os.path.join(correct_output_dir,dir)
        incorrect_target_sub_dir = os.path.join(incorrect_output_dir,dir)
        print(sub_dir)
        print(correct_target_sub_dir)
        print(incorrect_target_sub_dir)

        if not os.path.exists(correct_target_sub_dir):
            os.makedirs(correct_target_sub_dir)

        if not os.path.exists(incorrect_target_sub_dir):
            os.makedirs(incorrect_target_sub_dir)

        count_correct = 0
        count_incorrect = 0
        incorrect_counts = np.zeros((101,), dtype=np.int)

        for file in os.listdir(sub_dir):
            predict_class = get_label(os.path.join(sub_dir, file))
            if predict_class == i:
                count_correct += 1
                target_path = os.path.join(correct_target_sub_dir, file)
                shutil.copy(os.path.join(sub_dir, file),target_path)
            else:
                count_incorrect += 1
                incorrect_counts[predict_class] += 1
                target_path = os.path.join(incorrect_target_sub_dir, file)
                shutil.copy(os.path.join(sub_dir, file), target_path)

        for file in os.listdir(test_sub_dir):
            predict_class = get_label(os.path.join(test_sub_dir, file))
            if predict_class == i:
                count_correct += 1
                target_path = os.path.join(correct_target_sub_dir, file)
                shutil.copy(os.path.join(test_sub_dir, file),target_path)
            else:
                count_incorrect += 1
                incorrect_counts[predict_class] += 1
                target_path = os.path.join(incorrect_target_sub_dir, file)
                shutil.copy(os.path.join(test_sub_dir, file), target_path)

        print("count_correct:",count_correct)
        print("count_incorrect:", count_incorrect)

        print(incorrect_counts)
        mix_type = []
        for j in range(101):
            count = incorrect_counts[j]
            if (count + 0.0) / count_incorrect > 0.05:
                print(i,j,(count + 0.0) / count_incorrect)
                mix_type.append(j)
        mix_types.append(mix_type)
    print(mix_types)
    break