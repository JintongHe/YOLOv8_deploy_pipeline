# coding = utf8
import json
import math
import os

import numpy as np
import cv2


# 标签展示
def label2pic():
    # 读取图片
    images_path = r"E:\work\python\ai\data_handle\show\1050.jpg"
    label_path = r'E:\work\python\ai\data_handle\show\test.json'
    images_show_path = r'E:\work\python\ai\data_handle\show'
    image_name = images_path.split('\\')[-1]

    img = cv2.imread(images_path)
    # 读取标签
    f = open(label_path, encoding='utf8')
    contours = []
    for line in f:
        arr = line.replace('\n', '').split(' ')
        arr = [float(x) for x in arr]
        location = []
        # 读取四个点的坐标
        location.append([int(arr[0]), int(arr[1])])
        location.append([int(arr[2]), int(arr[3])])
        location.append([int(arr[4]), int(arr[5])])
        location.append([int(arr[6]), int(arr[7])])
        tmp_arr = np.array(location)
        contours.append(tmp_arr)

    # 标签打到图片上
    img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
    cv2.imwrite(os.path.join(images_show_path, image_name.replace('.jpg', "_contour" + '.jpg')), img_contour)


# 标签展示
def json_label2pic():
    # 读取图片
    images_path = r"E:\work\python\ai\data_handle\show\1050.jpg"
    label_path = r'E:\work\python\ai\data_handle\show\test.json'
    images_show_path = r'E:\work\python\ai\data_handle\show'
    image_name = images_path.split('\\')[-1]

    img = cv2.imread(images_path)
    # 读取标签
    f = open(label_path, encoding='utf8')
    res = ''
    for line in f:
        res = res + line
        res_json = json.loads(res)
        filename = res_json['filename']
        boxes = res_json['boxes']
        points = res_json['points']
        # print(res_json)
        contours = []
        for point in points:
            location = []
            location.append([int(point[0])-2, int(point[1])-2])
            location.append([int(point[0]) + 2, int(point[1]) - 2])
            location.append([int(point[0]) + 2, int(point[1]) + 2])
            location.append([int(point[0]) - 2, int(point[1]) + 2])
            tmp_arr = np.array(location)
            contours.append(tmp_arr)
        img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
        cv2.imwrite(os.path.join(images_show_path, image_name.replace('.jpg', "_contour" + '.jpg')), img_contour)
        break


if __name__ == '__main__':
    json_label2pic()
