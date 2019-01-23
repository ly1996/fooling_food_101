import re
from PIL import Image
from PIL import ImageEnhance
import glob as gb
import os

EXISTS = [
    'cheese_plate',
    'beignets',
    'hamburger',
    'tuna_tartare',
    'garlic_bread',
    'clam_chowder',
    'breakfast_burrito',
    'pancakes',
    'pork_chop',
    'beef_carpaccio',
    'baby_back_ribs',
    'hot_and_sour_soup',
    'prime_rib',
    'baklava',
    'apple_pie',
    'deviled_eggs',
    'bruschetta',
    'crab_cakes',
    'frozen_yogurt',
    'fried_calamari',
    'lasagna',
    'miso_soup',
    'risotto',
    'french_toast',
    'gyoza',
    'eggs_benedict',
    'beet_salad',
    'peking_duck',
    'dumplings',
    'caesar_salad'
]

original_dir = os.path.expanduser("../images")
dst_dir = os.path.expanduser("../train_set_new")
for dirnames in os.listdir(original_dir):
    if not dirnames in EXISTS:
        for dirs in os.listdir(original_dir + '/' + dirnames):
            path = original_dir + '/' + dirnames + '/' + dirs
            im = Image.open(path)
            # 亮度增强
            for i in [2,4]:
                brightness = 1 + 0.5 * i
                dst_bri_path = dst_dir + '/' + dirnames
                if not os.path.exists(dst_bri_path):
                    os.makedirs(dst_bri_path)        
                dst_bri_path = dst_bri_path + '/' + dirs[0:-4] + '_bright' + str(brightness) + '.jpg'
                enh_bri = ImageEnhance.Brightness(im)
                image_brightened = enh_bri.enhance(brightness)
                image_brightened.save(dst_bri_path, 'JPEG')
            # dst_flip_bri_path = 'image_data/' + result[0] + '_flip_bright' + str(brightness) + '.jpg'
            #     enh_bri = ImageEnhance.Brightness(dst_flip)
            #     image_brightened = enh_bri.enhance(brightness)
            #     image_brightened.save(dst_flip_bri_path, 'JPEG')

            # 色度增强
            #print('Brightness reinforcement complete.')
            for i in [2,4]:
                color = 1 + 0.5 * i
                dst_col_path = dst_dir + '/' + dirnames
                if not os.path.exists(dst_bri_path):
                    os.makedirs(dst_bri_path)
                dst_col_path = dst_col_path + '/' + dirs[0:-4] + '_color' + str(color) + '.jpg'
                #dst_col_path = dst_dir + result[0] + '_color' + str(color) + '.jpg'
                enh_col = ImageEnhance.Color(im)
                image_colored = enh_col.enhance(color)
                image_colored.save(dst_col_path, 'JPEG')
            #
            # 对比度增强
            #print('Contrast reinforcement complete.')
            for i in [2,4]:
                contrast = 1 + 0.5 * i
                dst_con_path = dst_dir + '/' + dirnames
                if not os.path.exists(dst_bri_path):
                    os.makedirs(dst_bri_path)
                dst_con_path = dst_con_path + '/' + dirs[0:-4] + '_contrast' + str(contrast) + '.jpg'
                #dst_con_path = dst_dir + result[0] + '_contrast' + str(contrast) + '.jpg'
                enh_con = ImageEnhance.Contrast(im)
                image_contrasted = enh_con.enhance(contrast)
                image_contrasted.save(dst_con_path, 'JPEG')
            #
            # 锐度增强
            #print('Sharpness reinforcement complete.')
            for i in [2,4]:
                sharpness = i
                dst_sha_path = dst_dir + '/' + dirnames
                if not os.path.exists(dst_sha_path):
                    os.makedirs(dst_sha_path)
                dst_sha_path = dst_sha_path + '/' + dirs[0:-4] + '_sharpness' + str(sharpness) + '.jpg'
                #dst_sha_path = dst_dir + result[0] + '_sharp' + str(sharpness) + '.jpg'
                enh_sha = ImageEnhance.Sharpness(im)
                image_sharped = enh_sha.enhance(sharpness)
                image_sharped.save(dst_sha_path, 'JPEG')
            #print('Sharpness reinforcement complete.')
    print(dirnames + ' complete')
