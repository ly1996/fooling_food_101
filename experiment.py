import os
from random import randint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np

normal_path = os.path.expanduser('~/winter-camp-pek/tmp/food-101/images')
final_data_set = os.path.expanduser('~/winter-camp-pek/food-101/food-101/new_images/final_data_set')

mix_list = [
    [2, 8, 42, 72, 100], [74, 77, 79, 93], [0, 8, 12, 22], [4, 5, 10, 13, 39, 99], [3, 5, 39, 99], [3, 39, 88, 99], [12, 31, 42], [5, 70, 75, 81, 88], [0, 42], [19, 56, 67, 96], [3, 5, 13
    , 99], [3, 5, 15, 48, 88], [2, 6, 22, 31], [3, 5, 10, 39, 48], [0, 17, 21, 98], [5, 48, 99], [17, 39], [21, 22, 73, 94, 98], [20, 60, 70, 81, 82], [9, 66, 96], [1, 38, 74, 77], [17, 22
    , 83, 98], [21, 58, 73, 98], [6, 21, 31, 68, 74, 100], [60], [49, 55, 80], [36, 37, 77, 87, 99], [8, 22, 39, 41, 60, 73], [34, 42, 49, 67], [14, 21, 22, 31, 58, 83], [5, 29, 34, 73, 87
    ], [6, 21, 29], [52, 92], [15, 71, 74, 75, 81, 86, 88, 92], [28, 67, 72], [37, 39, 41, 65], [9, 26, 31, 96], [39, 77, 79, 93], [1, 20, 40, 43, 68, 80, 85], [5, 22, 37, 77], [23, 38, 78
    , 92], [8, 24, 27, 54, 60, 81], [8, 72, 100], [18, 20, 68], [7, 18, 70, 71, 84], [22, 48, 58], [2, 42, 76], [82, 84, 87, 89], [5, 11, 13, 15], [0, 25, 42, 46], [37, 39, 77], [11, 15, 66,
     85, 96, 99], [0, 32, 74, 85, 92], [55, 80], [18, 41, 64, 75, 81], [53, 80, 96], [18, 67], [19, 84], [21, 22, 45], [8, 41, 82, 90], [18, 24, 27, 41], [15, 53, 55, 80], [18, 41, 47, 84],
    [21, 22, 29, 31, 58], [22, 32, 60, 73, 75, 81, 88], [5, 69], [51, 56, 96], [9, 42], [20, 23, 31, 40, 43], [65, 74], [18, 44, 90, 91], [44, 65, 70], [42, 74, 94, 100], [17, 22, 94,
    98], [1, 39, 77, 79, 92, 93], [18, 54, 81], [3, 10, 59], [1, 37, 39, 79, 93], [40, 47, 70, 74], [1, 37, 77, 93], [1, 53, 96], [18, 64, 75], [18, 47, 59, 84, 89], [14, 17, 21, 29, 94, 98],
    [44, 47, 71, 82, 89], [0, 2, 92], [15, 95, 99], [26, 39, 77, 89], [3, 5, 7, 15, 33, 70], [18, 47, 77, 82, 84, 87], [59, 70, 80, 81, 91], [70, 81, 82, 90], [2, 15, 85], [1, 37, 39,
    77, 79], [8, 17, 21, 73, 83, 100], [16, 86], [15, 80], [31, 70], [21, 22], [3, 4, 5, 15, 39], [0, 8, 21, 42, 72, 94]
    ]

def pick_pictures():
    rst = {}
    dirs = os.listdir(final_data_set)
    dirs.sort()
    x = randint(0, 21)
    class_exists=[]
    while (x == 16 or x == 17):
        x = randint(0, 21)
    #print(x, dirs[x])
    rst['label'] = dirs[x]
    rst['index'] = x
    class_exists.append(x)

    bad_img_dir = final_data_set + '/' + rst['label']
    img_list = []
    img_files = os.listdir(bad_img_dir)
    total = len(img_files)
    temp = []
    cnt = 0
    while cnt < 4:
        x = randint(0, total - 1)
        if not x in temp:
            temp.append(x)
            img_list.append([bad_img_dir + '/' + img_files[x], rst['label'] + '/' + img_files[x], rst['index']])
            cnt+=1
    #print(img_list)

    #temp = mix_list[rst['index']]
    if min(mix_list[rst['index']]) < 22:
        class_exists.append(min(mix_list[rst['index']]))
        mix_dir = final_data_set + '/' + dirs[min(mix_list[rst['index']])]
        img_files = os.listdir(mix_dir)
        total = len(img_files)
        temp = []
        cnt = 0
        while cnt < 2:
            x = randint(0, total - 1)
            if not x in temp:
                temp.append(x)
                img_list.append([mix_dir + '/' + img_files[x], dirs[min(mix_list[rst['index']])] + '/' + img_files[x], min(mix_list[rst['index']])])
                cnt+=1
    #print(rst['index'], mix_dir)
    #print(img_list)

    while len(img_list) < 8:
        x = randint(0, 21)
        if not x in class_exists:
            class_exists.append(x)
            new_dir = final_data_set + '/' + dirs[x]
            img_files = os.listdir(new_dir)
            total = len(img_files)
            x0 = randint(0, total - 1)
            img_list.append([new_dir + '/' + img_files[x0], dirs[x] + '/' + img_files[x0], x])

    #print(img_list)
    rst['img_list'] = img_list

    return rst

model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-54-1.1064.hdf5")
resnet_model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-new-55-1.0364.hdf5")
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
    label = np.argmax(pred)

    return label

normal_pass_count = 0
noise_pass_count = 0
for idx in range(1000):
    print(idx)
    pictures = pick_pictures()

    target_class = pictures['index']
    target_type_name = pictures['label']

    noise_label_list = []
    noise_index_list = []

    normal_label_list = []
    normal_index_list = []

    real_index_list = []

    img_list = pictures['img_list']

    is_normal_right = True
    is_noise_right = True

    for i in range(8):
        img_description = img_list[i]
    # for img_description in img_list:

        real_label = img_description[2]
        # noise_label = get_label(img_description[0])
        # normal_label = get_label(normal_path + '/' + img_description[1])

        noise_label = get_label_inception(img_description[0])
        normal_label = get_label_inception(normal_path + '/' + img_description[1])

        if noise_label == target_class:
            noise_label_list.append(img_description[0])
            noise_index_list.append(i)
        if normal_label == target_class:
            normal_label_list.append(normal_path + '/' + img_description[1])
            normal_index_list.append(i)

        if real_label == target_class:
            real_index_list.append(i)
            if noise_label != target_class:
                is_noise_right = False
            if normal_label != target_class:
                is_normal_right = False
        else:
            if noise_label == target_class:
                is_noise_right = False
            if normal_label == target_class:
                is_normal_right = False
    if is_normal_right:
        normal_pass_count += 1
    if is_noise_right:
        noise_pass_count += 1

    if len(noise_label_list) != 0 and is_normal_right :
        print(pictures)
        print("target class", target_class, real_index_list)
        print("target_type_name", target_type_name)
        print("is_normal_right", is_normal_right)
        print(normal_label_list, normal_index_list)
        print("is_noise_right", is_noise_right)
        print(noise_label_list, noise_index_list)

    # print("target class", target_class,real_index_list)
    # print("target_type_name", target_type_name)
    # print("is_normal_right",is_normal_right)
    # print(normal_label_list,normal_index_list)
    # print("is_noise_right", is_noise_right)
    # print(noise_label_list,noise_index_list)
print(normal_pass_count)
print(noise_pass_count)




