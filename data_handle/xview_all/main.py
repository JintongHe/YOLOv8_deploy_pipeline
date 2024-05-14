# coding=utf8
import os
from shutil import copyfile
import shutil

import numpy as np
import cv2
import random


# 获取数据
def get_data():
    path = r'E:\work\data\xview\dota-all\dota-xview\xview_split\data_with_YOLO_format_instance_segmentation'
    path1 = r'E:\work\data\xview_all/images'
    path2 = r'E:\work\data\xview_all/labels'
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    file_names = os.listdir(path)
    index = 0

    for file_name in file_names:
        index += 1
        if index % 100 == 0:
            print(index)
        if file_name.endswith('.png'):
            shutil.copy(path + '/' + file_name, path1 + '/' + file_name)
        elif file_name.endswith('.txt'):
            shutil.copy(path + '/' + file_name, path2 + '/' + file_name)


# 生成train、test、val数据
def get_train_test_val():
    path = r'E:\work\data\xview_all'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'E:\work\data\xview_all\train_val_test'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(target_path + '/images'):
        shutil.rmtree(target_path + '/images')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/labels'):
        shutil.rmtree(target_path + '/labels')  # 递归删除文件夹，即：删除非空文件夹
    os.makedirs(target_path + '/images')
    os.makedirs(target_path + '/labels')

    for data_type in ['train', 'test', 'val']:
        os.makedirs(target_path + '/images/' + data_type)
        os.makedirs(target_path + '/labels/' + data_type)

    sums = len(image_names)
    num = 0
    img_suff = '.png'
    for image_name in image_names:
        source = path + '/images/' + image_name
        num += 1
        if num % 100 == 0:
            print(num)

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= 0.7:
            target = target_path + '/images/train/' + image_name
        elif rand_rate <= 0.9:
            target = target_path + '/images/test/' + image_name
        else:
            target = target_path + '/images/val/' + image_name
        copyfile(source, target)


        image_label = image_name.replace(img_suff, '.txt')
        if image_label in image_labels:
            label_source = path + '/labels/' + image_label
            if rand_rate <= 0.7:
                label_target = target_path + '/labels/train/' + image_label
            elif rand_rate <= 0.9:
                label_target = target_path + '/labels/test/' + image_label
            else:
                label_target = target_path + '/labels/val/' + image_label

            copyfile(label_source, label_target)


# 色差处理
def color_handle():
    path = r'E:\work\data\xview_all/train_val_test'
    for data_type in ['train', 'val', 'test']:
        path1 = path + '/images/' + data_type
        path2 = path + '/labels/' + data_type

        img_names = os.listdir(path1)
        img_suff = '.png'

        index = 0

        for img_name in img_names:
            index += 1
            if index % 100 == 0:
                print(index)
            file1 = path1 + '/' + img_name
            file2 = path2 + '/' + img_name.replace(img_suff, '.txt')
            img = cv2.imread(file1)

            # ---------------------------------
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
            img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(file1.replace(img_suff, "_r1_" + img_suff), img2)
            shutil.copy(file2, file2.replace(".txt", "_r1_.txt"))
            # ---------------------------------
            """
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 2] = img_hsv[:, :, 2] * 0.5
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
            img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(file1.replace(img_suff, "_r2_" + img_suff), img1)
            shutil.copy(file2, file2.replace(".txt", "_r2_.txt"))
            """


# 高斯模糊、椒盐模糊
def gaussian_salt():
    path = r'E:\work\data\xview_all/train_val_test'
    path1 = path + '/images'
    path2 = path + '/labels'

    img_suff = '.png'

    for name in ['train', 'val', 'test']:
        path_image = path1 + '/' + name
        path_label = path2 + '/' + name
        image_names = os.listdir(path_image)
        label_names = os.listdir(path_label)
        index = 0
        for image_name in image_names:
            index += 1
            if index % 100 == 0:
                print(index)
            file1 = path_image + '/' + image_name
            file2 = path_label + '/' + image_name.replace(img_suff, '.txt')

            # 高斯模糊
            img = cv2.imread(file1)
            img1 = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)

            cv2.imwrite(file1.replace(img_suff, "_ga_" + img_suff), img1)
            shutil.copy(file2, file2.replace(".txt", "_ga_.txt"))
            # 椒盐模糊
            img_salt = cv2.imread(file1)
            noise = np.zeros(img_salt.shape, np.uint8)
            cv2.randu(noise, 0, 255)
            black = noise < 16
            white = noise > 255
            img_salt[black] = img_salt[white] = 40
            cv2.imwrite(file1.replace(img_suff, "_salt_" + img_suff), img_salt)
            shutil.copy(file2, file2.replace(".txt", "_salt_.txt"))


# 统计train\test\val的标签数据量
def count_data():
    cate_name_list = ['Building', 'Small-Car', 'Truck', 'Cargo-Truck', 'Damaged-Building', 'Trailer', 'Truck-w/Flatbed', 'Bus',
     'Passenger-Vehicle', 'Construction-Site', 'Utility-Truck', 'Maritime-Vessel', 'Motorboat', 'Sailboat',
     'Scraper/Tractor', 'Excavator', 'Dump-Truck', 'Shipping-Container', 'Front-loader/Bulldozer', 'Mobile-Crane',
     'Crane-Truck', 'Truck-w/Box', 'Truck-Tractor', 'Vehicle-Lot', 'Pickup-Truck', 'Cement-Mixer',
     'Engineering-Vehicle', 'Storage-Tank', 'Ground-Grader', 'Hut/Tent', 'Facility', 'Fixed-wing-Aircraft',
     'Reach-Stacker', 'Tower', 'Passenger-Car', 'Cargo-Car', 'Shed', 'Cargo-Plane', 'Shipping-container-lot',
     'Locomotive', 'Pylon', 'Tank-car', 'Flat-Car', 'other', 'Railway-Vehicle', 'Fishing-Vessel', 'Barge', 'Tugboat',
     'Haul-Truck', 'Helipad', 'Tower-crane', 'Container-Crane', 'Small-Aircraft', 'Helicopter', 'Oil-Tanker', 'Ferry',
     'Truck-w/Liquid', 'Aircraft-Hangar', 'Yacht', 'Container-Ship', 'Straddle-Carrier']

    path = r'E:\work\data\xview_all/train_val_test/labels'
    data_types = os.listdir(path)
    for data_type in data_types:
        path1 = path + '/' + data_type
        label_names = os.listdir(path1)
        dicts = {}
        print(data_type)
        print(len(label_names))
        index = 0
        for label_name in label_names:
            index += 1
            if index % 100000 == 0:
                print(index)
            f = open(path1 + '/' + label_name, encoding='utf8')
            for line in f:
                arr = line.replace('\n', '').split(' ')
                cate_id = arr[0]
                cate_name = cate_name_list[int(cate_id)]
                if cate_name not in dicts.keys():
                    dicts[cate_name] = 1
                else:
                    dicts[cate_name] += 1
            f.close()
        print(sorted(dicts.keys()))
        print(dicts)


if __name__ == '__main__':
    count_data()
