# coding=utf8
import os
from shutil import copyfile
import shutil

import numpy as np
import cv2
import random
# 处理xview数据，主要是yolo格式


# 过滤车子数据
def filter_car():
    path = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation'
    namelist = os.listdir(path)
    car_list = ['1', '2', '3', '6', '7', '8', '10', '16', '20', '21', '24', '25', '26', '34', '35', '39', '41', '42', '44', '48', '56']
    print(len(namelist))
    index = 0
    for name in namelist:
        if 'txt' not in name:
            continue
        index += 1
        if index % 100 == 0:
            print(index)
        f = open(path + '/' + name, encoding='utf8')
        path1 = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation_car/images'
        path2 = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation_car/labels'
        res = []
        for line in f:
            cate = line.split(' ')[0]
            if cate in car_list:
                res.append(line)

        if len(res) == 0:
            print(name)
        else:
            out = open(path2 + '/' + name, 'w', encoding='utf8')
            out.write(''.join(res))
            out.close()
            # 复制图片
            copyfile(path + '/' + name.replace('txt', 'png'), path1 + '/' + name.replace('txt', 'png'))


# 色差处理
def color_handle():
    path1 = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation_car/images'
    path2 = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation_car/labels'

    img_names = os.listdir(path1)
    img_suff = '.png'

    index = 0

    for name in img_names:
        index += 1
        if index % 100 == 0:
            print(index)
        file1 = path1 + '/' + name
        file2 = path2 + '/' + name
        img = cv2.imread(file1)

        # ---------------------------------
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
        img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(file1.replace(img_suff, "_r1_" + img_suff), img2)
        shutil.copy(file2.replace(img_suff, ".txt"), file2.replace(img_suff, ".txt").replace(".txt", "_r1_.txt"))
        # ---------------------------------
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] * 0.5
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
        img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(file1.replace(img_suff, "_r2_" + img_suff), img1)
        shutil.copy(file2.replace(img_suff, ".txt"), file2.replace(img_suff, ".txt").replace(".txt", "_r2_.txt"))


# 生成train、test、val数据
def get_train_test_val(path):
    path = 'E:/work/data/xview/dota-all/dota-xview/xview_split/data_with_YOLO_format_instance_segmentation_car'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = 'E:/work/python/ai/car/data1'
    if os.path.exists(target_path + '/images'):
        shutil.rmtree(target_path + '/images')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/labels'):
        shutil.rmtree(target_path + '/labels')  # 递归删除文件夹，即：删除非空文件夹
    os.makedirs(target_path + '/images')
    os.makedirs(target_path + '/labels')

    for name in ['train', 'test', 'val']:
        os.makedirs(target_path + '/images/' + name)
        os.makedirs(target_path + '/labels/' + name)

    sums = int(len(image_names)/3)
    num = 0
    for name in image_names:
        if '_r' in name:
            continue
        source = path + '/images/' + name
        source1 = path + '/images/' + name.replace('.png', '_r1_.png')
        num += 1
        if num % 100 == 0:
            print(num)

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= 0.7:
            target = target_path + '/images/train/' + name
            target1 = target_path + '/images/train/' + name.replace('.png', '_r1_.png')
        elif rand_rate <= 0.9:
            target = target_path + '/images/test/' + name
            target1 = target_path + '/images/test/' + name.replace('.png', '_r1_.png')
        else:
            target = target_path + '/images/val/' + name
            target1 = target_path + '/images/val/' + name.replace('.png', '_r1_.png')
        copyfile(source, target)
        copyfile(source1, target1)

        if name.replace('.png', '.txt') in image_labels:
            source = path + '/labels/' + name.replace('.png', '.txt')
            source1 = path + '/labels/' + name.replace('.png', '_r1_.txt')
            if rand_rate <= 0.7:
                target = target_path + '/labels/train/' + name.replace('.png', '.txt')
                target1 = target_path + '/labels/train/' + name.replace('.png', '_r1_.txt')
            elif rand_rate <= 0.9:
                target = target_path + '/labels/test/' + name.replace('.png', '.txt')
                target1 = target_path + '/labels/test/' + name.replace('.png', '_r1_.txt')
            else:
                target = target_path + '/labels/val/' + name.replace('.png', '.txt')
                target1 = target_path + '/labels/val/' + name.replace('.png', '_r1_.txt')

            copyfile(source, target)
            copyfile(source1, target1)


# 变换车子序号
def change_num():
    car_list = ['1', '2', '3', '6', '7', '8', '10', '16', '20', '21', '24', '25', '26', '34', '35', '39', '41', '42', '44', '48', '56']
    num_dict = {}
    num_dict1 = {}

    for index in range(len(car_list)):
        key = car_list[index]
        value = index
        num_dict[key] = value
        num_dict1[key] = 0
    print(num_dict)
    path = 'E:/work/python/ai/car/data1/labels'
    for name in ['train', 'test', 'val']:
        image_labels = os.listdir(path + '/' + name + ' - 副本')
        for image_label in image_labels:
            f = open(path + '/' + name + ' - 副本' + '/' + image_label, encoding='utf8')
            out = open(path + '/' + name + '/' + image_label, 'w', encoding='utf8')
            for line in f:
                key = line.split(' ')[0]
                value = num_dict[key]
                out.write(str(value) + ' ' + ' '.join(line.split(' ')[1:]))
            f.close()
            out.close()


# 高斯模糊、椒盐模糊
def gaussian_salt():
    path = 'E:/work/python/ai/car/data1'
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
            file2 = path_label + '/' + image_name
            if '_ga_' in file1 or '_salt_' in file1:
                os.remove(file1)
                continue

            # 高斯模糊
            img = cv2.imread(file1)
            img_ = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)

            cv2.imwrite(file1.replace(img_suff, "_ga_" + img_suff), img_)
            shutil.copy(file2.replace(img_suff, ".txt"), file2.replace(img_suff, ".txt").replace(".txt", "_ga_.txt"))
            # 椒盐模糊
            img_salt = cv2.imread(file1)
            noise = np.zeros(img_salt.shape, np.uint8)
            cv2.randu(noise, 0, 255)
            black = noise < 16
            white = noise > 255
            img_salt[black] = img_salt[white] = 40
            cv2.imwrite(file1.replace(img_suff, "_salt_" + img_suff), img_salt)
            shutil.copy(file2.replace(img_suff, ".txt"), file2.replace(img_suff, ".txt").replace(".txt", "_salt_.txt"))


if __name__ == '__main__':
    gaussian_salt()
