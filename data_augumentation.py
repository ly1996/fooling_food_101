#!/usr/bin/env python
#-*- coding: utf-8 -*-

# import packages
import os
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

original_dir = os.path.expanduser("../images")
dst_dir = os.path.expanduser("../train_set_new")
for dirnames in os.listdir(original_dir):
    for dirs in os.listdir(original_dir + '/' + dirnames):
        #result = re.findall(r"(.*).jpg", file)
        #number = result[0]
        #img = load_img(os.path.join(original_dir, file))
        path = original_dir + '/' + dirnames + '/' + dirs
        img = load_img(path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        save_path = dst_dir + '/' + dirnames
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        i = 0
        for batch in datagen.flow(x,
                                batch_size=1,
                                save_to_dir=os.path.expanduser(save_path),
                                save_prefix=dirs[0:-4],
                                save_format='jpg'):
            i += 1
            if i >= 4:
                break  # otherwise the generator would loop indefinitely
    print(dirnames + ' complete')

# img = load_img(os.path.expanduser('~/Disk/ic-data/1.jpg'))  # this is a PIL image, please replace to your own file path
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
#
# i = 0
# for batch in datagen.flow(x,
# 						  batch_size=1,
#                           save_to_dir=os.path.expanduser('~/Disk/ic-data/aug'),
#                           save_prefix='1',
#                           save_format='jpg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely
