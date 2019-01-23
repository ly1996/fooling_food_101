import os
from tensorflow.keras.models import load_model
from PIL import Image

input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images")
model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-12-1.7504.hdf5")
model = load_model(model_path)

# def is_correct(model,img,i):
#

for root,dirs,files in os.walk(input_dir):
    dirs.sort()
    print(dirs)
    for i in range(len(dirs)):
        dir = dirs[i]
        sub_dir = os.path.join(input_dir,dir)
        print (sub_dir)
        for file in os.listdir(sub_dir):
            # print(file)
            img = Image.open(os.path.join(sub_dir, file))
    break