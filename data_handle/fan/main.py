# coding = utf8
import json
import os
from shutil import copyfile
import json
import math
import os
import shutil

import cv2
from tqdm import tqdm
import random
import numpy as np


# 生成train、test、val数据
def get_train_test_val():
    path = r'/home/zkxq/data/fan'
    image_names = os.listdir(path + '/images')
    label_names = os.listdir(path + '/labels')

    print(len(image_names))
    print(len(label_names))
    # 创建相关文件夹
    target_path = r'/home/zkxq/project/caoshouhong/ultralytics/data/fan'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(target_path + '/images'):
        shutil.rmtree(target_path + '/images')  # 递归删除文件夹，即：删除非空文件夹
    if os.path.exists(target_path + '/labels'):
        shutil.rmtree(target_path + '/labels')  # 递归删除文件夹，即：删除非空文件夹

    os.makedirs(target_path + '/images')
    os.makedirs(target_path + '/labels')

    for data_type in ['train', 'val']:
        os.makedirs(target_path + '/images/' + data_type)
        os.makedirs(target_path + '/labels/' + data_type)

    sums = len(image_names)
    img_suff = '.jpg'
    for image_name in tqdm(image_names):
        label_name = image_name.replace(img_suff, '.txt')
        if label_name not in label_names:
            print(image_name)
            continue
        image_source = path + '/images/' + image_name
        label_source = path + '/labels/' + label_name

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= 0.9:
            image_target = target_path + '/images/train/' + image_name
            label_target = target_path + '/labels/train/' + label_name
        else:
            image_target = target_path + '/images/val/' + image_name
            label_target = target_path + '/labels/val/' + label_name
        shutil.copyfile(image_source, image_target)
        shutil.copyfile(label_source, label_target)


# 处理标注的一些风机数据
def handle_mark_fan():
    path = r'E:\work\data\fan\风力发电厂项目图像训练集'
    file_names = os.listdir(path)
    for file_name in file_names:
        print(file_name)
        image_label_names = os.listdir(path + '/' + file_name)
        image_names = []
        label_names = []
        for name in image_label_names:
            if '.png' in name:
                image_names.append(name)
            elif '.json' in name:
                label_names.append(name)
        for label_name in label_names:
            print(file_name + '/' + label_name)
            f = open(path + '/' + file_name + '/' + label_name, encoding='utf8')
            res = ''
            for line in f:
                res = res + line
            f.close()
            res_json = json.loads(res)

            for res_data in res_json:
                out_res = ''
                file_upload = res_data['file_upload']
                source_image_name = file_upload.split('-')[1]
                target_image_name = file_upload
                annotations = res_data['annotations']
                if len(annotations) > 1:
                    print('annotations', annotations)
                annotation = annotations[0]
                results = annotation['result']
                for result in results:
                    original_width = result['original_width']
                    original_height = result['original_height']
                    value = result['value']
                    rectanglelabels = value['rectanglelabels']
                    rotation = value['rotation']
                    if len(rectanglelabels) > 1:
                        print('rectanglelabels', rectanglelabels)
                    if rotation != 0:
                        print('rotation', rotation)
                    rectanglelabel = rectanglelabels[0]
                    if rectanglelabel != 'Wind Turbines Foundation' and rectanglelabel != 'Wind Turbine Foundation':
                        print('rectanglelabel', rectanglelabel)
                        continue
                    x = value['x']/original_width
                    y = value['y']/original_height
                    width = value['width']/original_width
                    height = value['height']/original_height
                    out_res = out_res + '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'

                if out_res != '':
                    # 写入图像
                    source_image_path = path + '/' + file_name + '/' + source_image_name
                    target_image_path = r'E:\work\data\fan\images1' + '/' + target_image_name
                    copyfile(source_image_path, target_image_path)
                    # 写入标签
                    target_label_name = target_image_name.replace('.png', '.txt')
                    target_label_path = r'E:\work\data\fan\labels1' + '/' + target_label_name
                    out = open(target_label_path, 'w', encoding='utf8')
                    out.write(out_res)
                    out.close()
            break



if __name__ == '__main__':
    handle_mark_fan()
