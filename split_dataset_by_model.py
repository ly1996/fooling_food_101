import os

input_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/images")
model_path = os.path.expanduser("~/winter-camp-pek/tmp/fooling_food_101/checkpoint-12-1.7504.hdf5")

for root,dirs,files in os.walk(input_dir):
    print(dirs)