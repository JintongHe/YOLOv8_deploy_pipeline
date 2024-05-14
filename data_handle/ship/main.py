# coding = utf8
import json
import math
import os
import shutil

import cv2
from tqdm import tqdm
import random
import numpy as np


# 转换为yolo格式的label
def label2yolo():
    label_path = '/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/labels'
    image_path = '/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/images'
    label_yolo_path = '/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/labels_yolo'
    label_names = os.listdir(label_path)
    for label_name in tqdm(label_names):
        image_name = label_name.replace('.txt', '.png')
        image_detail_path = image_path + '/' + image_name
        img = cv2.imread(os.path.join(image_detail_path))
        w = img.shape[1]
        h = img.shape[0]
        f = open(label_path + '/' + label_name, encoding='utf8')
        out = open(label_yolo_path + '/' + label_name, 'w', encoding='utf8')
        for line in f:
            arr = line.replace('\n', '').split(' ')
            label = '0'
            yolo_arr = [label]
            for index in range(0, len(arr)):
                if index % 2 == 0:
                    yolo_arr.append(float(arr[index]) / w)
                else:
                    yolo_arr.append(float(arr[index]) / h)
            yolo_arr = [str(x) for x in yolo_arr]
            out.write(' '.join(yolo_arr) + '\n')
        f.close()
        out.close()


def color_handle():
    image_path = r'/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/20230915/images_split'
    label_path = r'/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/20230915/labels_yolo_split'
    image_names = os.listdir(image_path)
    img_suff = '.png'
    for image_name in tqdm(image_names):
        label_name = image_name.replace(img_suff, '.txt')
        image_detail_path = image_path + '/' + image_name
        label_detail_path = label_path + '/' + label_name
        img = cv2.imread(image_detail_path)

        # ---------------------------------
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
        img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(image_detail_path.replace(img_suff, "_r1_" + img_suff), img2)
        shutil.copy(label_detail_path, label_detail_path.replace(".txt", "_r1_.txt"))


# 生成或新增train、test、val数据
def get_train_test_val():
    path = r'/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901/mark_data/20230915'
    image_names = os.listdir(path + '/images_split')
    image_labels = os.listdir(path + '/labels_yolo_split')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'/home/zkxq/project/caoshouhong/ultralytics/data/ship20230901'

    sums = len(image_names)
    img_suff = '.png'
    for image_name in tqdm(image_names):
        label_name = image_name.replace(img_suff, '.txt')
        image_source = path + '/images_split/' + image_name
        label_source = path + '/labels_yolo_split/' + label_name

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums
        # 去除大文件
        image_size = os.stat(image_source).st_size / 1024 / 1024
        if image_size > 20:
            continue

        if rand_rate <= 0.8:
            image_target = target_path + '/images/train/' + image_name
            label_target = target_path + '/labels/train/' + label_name
        elif rand_rate <= 0.9:
            image_target = target_path + '/images/test/' + image_name
            label_target = target_path + '/labels/test/' + label_name
        else:
            image_target = target_path + '/images/val/' + image_name
            label_target = target_path + '/labels/val/' + label_name

        shutil.copyfile(image_source, image_target)
        shutil.copyfile(label_source, label_target)


if __name__ == '__main__':
    get_train_test_val()
