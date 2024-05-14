# coding = utf8
import json
import math
import os
import shutil

import cv2
from tqdm import tqdm
import random
import numpy as np


# 根据标签复制图片
def copy_images():
    label_path = r'/home/zkxq/data/car20230720/labels'
    image_source_path = '/home/zkxq/data/car20230705/images'
    image_target_path = '/home/zkxq/data/car20230720/images'
    label_names = os.listdir(label_path)
    for label_name in tqdm(label_names):
        image_name = label_name.replace('.txt', '.png')
        image_source = image_source_path + '/' + image_name
        image_target = image_target_path + '/' + image_name
        shutil.copyfile(image_source, image_target)


# 生成train、test、val数据
def get_train_test_val():
    path = r'/home/zkxq/data/car20230720'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'/home/zkxq/data/car20230720/train_val_test'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(target_path + '/images'):
        shutil.rmtree(target_path + '/images')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/labels'):
        shutil.rmtree(target_path + '/labels')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/images_large'):
        shutil.rmtree(target_path + '/images_large')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/labels_large'):
        shutil.rmtree(target_path + '/labels_large')  # 递归删除文件夹，即：删除非空文件夹
    os.makedirs(target_path + '/images')
    os.makedirs(target_path + '/labels')
    os.makedirs(target_path + '/images_large')
    os.makedirs(target_path + '/labels_large')

    for data_type in ['train', 'test', 'val']:
        os.makedirs(target_path + '/images/' + data_type)
        os.makedirs(target_path + '/labels/' + data_type)

    sums = len(image_names)
    img_suff = '.png'
    for image_name in tqdm(image_names):
        label_name = image_name.replace(img_suff, '.txt')
        image_source = path + '/images/' + image_name
        label_source = path + '/labels/' + label_name

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= 0.8:
            image_target = target_path + '/images/train/' + image_name
            label_target = target_path + '/labels/train/' + label_name
        elif rand_rate <= 0.9:
            image_target = target_path + '/images/test/' + image_name
            label_target = target_path + '/labels/test/' + label_name
        else:
            image_target = target_path + '/images/val/' + image_name
            label_target = target_path + '/labels/val/' + label_name
        # 去掉文件大于50M的数据
        image_size = os.stat(image_source).st_size / 1024 / 1024
        if image_size > 20:
            image_target = target_path + '/images_large/' + image_name
            label_target = target_path + '/labels_large/' + label_name

        shutil.copyfile(image_source, image_target)
        shutil.copyfile(label_source, label_target)


def getPolygonArea(points):
    '''
    brief: calculate the Polygon Area with vertex coordinates
    refer: https://blog.csdn.net/qq_38862691/article/details/87886871
    :param points: list, input vertex coordinates
    :return: float, polygon area
    '''

    sizep = len(points)
    if sizep<3:
        return 0.0

    area = points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
    for i in range(1, sizep):
        v = i - 1
        area += (points[v][0] * points[i][1])
        area -= (points[i][0] * points[v][1])

    return abs(0.5 * area)


# 生成codo数据格式
def create_coco_data():
    image_id = 1
    label_id = 1
    for file_name in ['train', 'test', 'val']:
        out = open('/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/annotations/' + file_name + '20230720.json', 'w', encoding='utf8')
        res_json = {'images': [], 'annotations': [], 'categories': []}
        image_path = '/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/images/' + file_name
        image_names = os.listdir(image_path)
        image_names_arr = np.arange(len(image_names))
        np.random.shuffle(image_names_arr)

        for index in tqdm(image_names_arr):
            # image
            image_name = image_names[index]
            img = cv2.imread(image_path + '/' + image_name)
            image_json = {
                'id': image_id,
                'file_name': image_name,
                'height': img.shape[0],
                'width': img.shape[1]
            }
            image_id += 1
            res_json['images'].append(image_json)
            # annotation
            label_name = image_name.replace('.png', '.txt')
            label_f = open('/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/labels/' + file_name + '/' + label_name, encoding='utf8')
            for line in label_f:
                arr = line.replace('\n', '').split(' ')
                arr = [int(x.split('.')[0]) for x in arr]
                category_id = arr[-1] + 1
                iscrowd = 0
                points = [[arr[0], arr[1]], [arr[2], arr[3]], [arr[4], arr[5]], [arr[6], arr[7]]]
                # area = getPolygonArea(points)
                area = 0
                segmentation = arr[0:8]
                min_x = min(arr[0], arr[2], arr[4], arr[6])
                min_y = min(arr[1], arr[3], arr[5], arr[7])
                max_x = max(arr[0], arr[2], arr[4], arr[6])
                max_y = max(arr[1], arr[3], arr[5], arr[7])
                bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
                annotation = {
                    'id': label_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'segmentation': [segmentation],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': iscrowd
                }
                label_id += 1
                res_json['annotations'].append(annotation)
            label_f.close()
        # categories
        res_json['categories'].append({'supercategory': 'small_car', 'id': 1, 'name': 'small_car'})
        res_json['categories'].append({'supercategory': 'large_car', 'id': 2, 'name': 'large_car'})
        out.write(str(res_json).replace('\'', '"'))
        out.close()


def test():
    f = open('/home/zkxq/project/caoshouhong/mmdetection/data/car20230705/annotations/val.json', encoding='utf8')
    index = 0
    for line in f:
        print(index)
        index += 1
        res_json = json.loads(line)
        print(res_json.keys())


def get_file_size():
    file = '/home/zkxq/project/caoshouhong/mmdetection/data/car20230705/test/1457__1800_1800.png'
    size = os.stat(file).st_size/1024/1024
    print(f'The size of {file} is {size} m.')


# 转换为yolo格式的label
def label2yolo():
    for file_name in ['train', 'test', 'val']:
        label_path = '/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/labels/' + file_name
        label_names = os.listdir(label_path)
        print(file_name)
        for label_name in tqdm(label_names):
            image_name = label_name.replace('.txt', '.png')
            image_detail_path = '/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/images/' + file_name + '/' + image_name
            img = cv2.imread(os.path.join(image_detail_path))
            w = img.shape[1]
            h = img.shape[0]
            f = open('/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/labels/' + file_name + '/' + label_name, encoding='utf8')
            out = open('/home/zkxq/project/caoshouhong/mmdetection/data/car20230720/labels_yolo/' + file_name + '/' + label_name, 'w', encoding='utf8')
            for line in f:
                arr = line.replace('\n', '').split(' ')
                label = arr[-1]
                yolo_arr = [label]
                for index in range(len(arr)-1):
                    if index % 2 == 0:
                        yolo_arr.append(float(arr[index]) / w)
                    else:
                        yolo_arr.append(float(arr[index]) / h)
                yolo_arr = [str(x) for x in yolo_arr]
                out.write(' '.join(yolo_arr) + '\n')
            f.close()
            out.close()


# 转换为yolo格式的label
def label2yolo():
    label_path = '/home/zkxq/project/caoshouhong/ultralytics/data/car20230809/mark_car/labels'
    image_path = '/home/zkxq/project/caoshouhong/ultralytics/data/car20230809/mark_car/images'
    label_yolo_path = '/home/zkxq/project/caoshouhong/ultralytics/data/car20230809/mark_car/labels_yolo'
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
            label = arr[0]
            yolo_arr = [label]
            for index in range(1, len(arr)):
                if index % 2 == 1:
                    yolo_arr.append(float(arr[index]) / w)
                else:
                    yolo_arr.append(float(arr[index]) / h)
            yolo_arr = [str(x) for x in yolo_arr]
            out.write(' '.join(yolo_arr) + '\n')
        f.close()
        out.close()


# 生成train、test、val数据
def get_train_test_val():
    path = r'/home/zkxq/project/caoshouhong/ultralytics/data/car20230809/mark_car'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels_yolo')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'/home/zkxq/project/caoshouhong/ultralytics/data/car20230809'

    sums = len(image_names)
    img_suff = '.png'
    for image_name in tqdm(image_names):
        label_name = image_name.replace(img_suff, '.txt')
        image_source = path + '/images/' + image_name
        label_source = path + '/labels_yolo/' + label_name

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

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
