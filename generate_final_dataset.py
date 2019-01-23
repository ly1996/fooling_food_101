import os
import shutil
import random

noise_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/correct_with_noise")
incorrect_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/incorrect")
output_dir = os.path.expanduser("~/winter-camp-pek/food-101/food-101/new_images/final_data_set")

for root,dirs,files in os.walk(noise_dir):
    dirs.sort()
    print(dirs)

    for i in range(len(dirs)):
        dir = dirs[i]
        sub_dir = os.path.join(noise_dir, dir)
        incorrect_sub_dir = os.path.join(incorrect_dir, dir)
        target_sub_dir = os.path.join(output_dir, dir)

        if not os.path.exists(target_sub_dir):
            os.makedirs(target_sub_dir)

        print(sub_dir)
        print(incorrect_sub_dir)
        print(target_sub_dir)

        noise_files = os.listdir(sub_dir)
        noise_files_len = len(noise_files)

        incorrect_files = os.listdir(incorrect_sub_dir)
        incorrect_files_len = len(incorrect_files)
        print(incorrect_files_len)

        for file in noise_files:
            target_path = os.path.join(target_sub_dir, file)
            shutil.copy(os.path.join(sub_dir, file), target_path)

        count = int(noise_files_len * (1000 - incorrect_files_len) / (incorrect_files_len + 0.0))
        if count == 0:
            count = 1
        print(count)

        tmp = 0
        paths = []
        while True:
            idx = random.randint(0,incorrect_files_len-1)
            file = incorrect_files[idx]
            if file not in paths:
                paths.append(file)
                tmp += 1
            if tmp == count:
                break
        print(len(paths))

        for file in paths:
            target_path = os.path.join(target_sub_dir, file)
            shutil.copy(os.path.join(incorrect_sub_dir, file), target_path)
    break