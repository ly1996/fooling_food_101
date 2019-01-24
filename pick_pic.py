import os
from random import randint

final_data_set = os.path.expanduser('~/winter-camp-pek/food-101/food-101/new_images/final_data_set')
#final_data_set = 'E:/饶世杰的文件/2019谷歌冬令营/food-101/images'

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

print(pick_pictures())

