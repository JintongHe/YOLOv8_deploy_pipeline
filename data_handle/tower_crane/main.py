# coding = utf8
import json
import os
from shutil import copyfile
import shutil

import numpy as np
import cv2
import random


# 生成yolo格式的标签数据
def generate():
    data_path = r'E:\work\data\tower_crane\source'
    out_path = r'E:\work\data\tower_crane\out'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(out_path + '/images'):
        os.makedirs(out_path + '/images')
    if not os.path.exists(out_path + '/labels'):
        os.makedirs(out_path + '/labels')
    file_names = os.listdir(data_path)
    print(file_names)
    for file_name in file_names:
        file_path = data_path + '/' + file_name
        images_path = file_path + '/images'
        labels_path = file_path + '/annfiles'

        image_names = os.listdir(images_path)
        label_names = os.listdir(labels_path)
        print('image_names个数', len(image_names))
        index = 0
        for image_name in image_names:
            index += 1
            if index % 100 == 0:
                print('index', index)
            image_source = images_path + '/' + image_name
            image_target = out_path + '/images/' + image_name
            '''生成yolo标签'''
            img = cv2.imread(image_source)
            width = img.shape[1]
            height = img.shape[0]
            # 读取对应的txt标签文件
            label_name = image_name.split('.')[0] + '.' + 'txt'
            if label_name not in label_names:
                continue
            else:
                f = open(labels_path + '/' + label_name, encoding='utf8')
                res_all = []
                for line in f:
                    arr = line.strip().split(" ")
                    if len(arr) < 5:
                        continue
                    else:

                        if arr[-2].lower() != 'tower-crane' and arr[-1].lower() != 'tower-crane':
                            continue
                        res_arr = arr[:8]
                        cate_id = '0'
                        res_arr.insert(0, cate_id)

                        if file_name == 'trainval':
                            res_arr[3] = float(res_arr[3]) - float(res_arr[1])
                            res_arr[5] = float(res_arr[5]) - float(res_arr[1])
                            res_arr[6] = float(res_arr[6]) - float(res_arr[2])
                            res_arr[8] = float(res_arr[8]) - float(res_arr[2])
                            for i in range(len(res_arr)):
                                if i % 2 == 1:
                                    res_arr[i] = float(res_arr[i]) / width
                                    res_arr[i + 1] = float(res_arr[i + 1]) / height
                            res_arr = [str(x) for x in res_arr]
                        else:
                            xd = (float(res_arr[3]) - float(res_arr[1]))/2
                            yd = (float(res_arr[8]) - float(res_arr[2]))/2
                            for i in range(len(res_arr)):
                                if i % 2 == 1:
                                    print(xd, yd)
                                    res_arr[i] = (float(res_arr[i]) + 50) / width
                                    res_arr[i + 1] = (float(res_arr[i + 1]) + 50) / height
                            res_arr = [str(x) for x in res_arr]

                        res = " ".join(res_arr) + "\n"
                        res_all.append(res)
                if len(res_all) > 0:
                    # copyfile(image_source, image_target)
                    out = open(out_path + '/labels/' + label_name, 'w', encoding='utf8')
                    for res in res_all:
                        out.write(res)
                    out.close()
                f.close()


# 在图片中显示数据
def label2pic():
    from tqdm import tqdm
    images_path = r"E:\work\data\tower_crane\train_val_test\images\train"
    labels_path = r"E:\work\data\tower_crane\train_val_test\labels\train"
    images_show_path = r"E:\work\data\tower_crane\train_val_test\labels\show"
    if not os.path.exists(images_show_path):
        os.makedirs(images_show_path)
    files = os.listdir(images_path)
    img_format = '.png'
    img_list = [x for x in files if x.endswith(img_format)]

    class_label = ['tower-crane']

    for item in tqdm(img_list):
        img = cv2.imread(os.path.join(images_path, item))
        w = img.shape[1]
        h = img.shape[0]

        contours = []
        if not os.path.exists(os.path.join(labels_path, item.replace(img_format, ".txt"))):
            continue
        with open(os.path.join(labels_path, item.replace(img_format, ".txt")), "r") as t:
            s_label = []
            f = t.readlines()
            for line in f:
                arr = line.strip().split(" ")
                contours.append(arr[1:])
                s_label.append(arr[0])

        contours_tmp = []
        length_max = 0

        label_position = []

        for contour in contours:
            length = len(contour)
            if int(length / 2) > length_max:
                length_max = int(length / 2)

            tmp_position = []
            contour_tmp = []
            for i in range(len(contour)):
                if i % 2 == 0:
                    tmp = list()
                    tmp_x = int(float(contour[i]) * w)
                    tmp_y = int(float(contour[i + 1]) * h)
                    tmp.append([tmp_x, tmp_y])
                    contour_tmp.append(tmp)

                    if len(tmp_position) == 0:
                        tmp_position = [tmp_x, tmp_y]
                    else:
                        if tmp_position[0] > tmp_x:
                            tmp_position[0] = tmp_x
                            tmp_position[1] = tmp_y
                        else:
                            pass
            label_position.append(tuple(tmp_position))
            contours_tmp.append(contour_tmp)

        for i in range(len(contours_tmp)):
            if len(contours_tmp[i]) < length_max:
                s = contours_tmp[i][-1]
                for j in range(length_max - len(contours_tmp[i])):
                    contours_tmp[i].append(s)
            contours_tmp[i] = np.array(contours_tmp[i])

        contours_tmp = tuple(contours_tmp)

        img_contour = cv2.drawContours(img, contours_tmp, -1, (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        for num in range(len(label_position)):
            img_contour = cv2.putText(img_contour, class_label[int(s_label[num])], label_position[num], font, 0.4, color, 1)
        cv2.imwrite(os.path.join(images_show_path, item.replace(img_format, "_contour" + img_format)), img_contour)


# 生成train、test、val数据
def get_train_test_val():
    path = r'E:\work\data\tower_crane\out'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')
    image_labelxs = os.listdir(path + '/labelsx')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'E:\work\data\tower_crane\train_val_test'
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

    sums = 0
    image_names_x = []
    for image_name in image_names:
        if 'x' not in image_name:
            image_names_x.append(image_name)
            sums += 1
        elif image_name.split('.')[0] + '.json' in image_labelxs:
            image_names_x.append(image_name)
            sums += 1

    num = 0
    img_suff = '.png'
    for image_name in image_names_x:
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

        if 'x' not in image_name:
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
        else:
            # 处理labelme标注的标签
            image_label = image_name.replace(img_suff, '.json')
            label_source = path + '/labelsx/' + image_label
            if rand_rate <= 0.7:
                f = open(label_source, encoding='utf8')
                out = open(target_path + '/labels/train/' + image_label.replace('.json', '.txt'), 'w', encoding='utf8')
            elif rand_rate <= 0.9:
                f = open(label_source, encoding='utf8')
                out = open(target_path + '/labels/test/' + image_label.replace('.json', '.txt'), 'w', encoding='utf8')
            else:
                f = open(label_source, encoding='utf8')
                out = open(target_path + '/labels/val/' + image_label.replace('.json', '.txt'), 'w', encoding='utf8')
            res = ''
            for line in f:
                res = res + line
            res_json = json.loads(res)
            shapes = res_json['shapes']
            imageHeight = int(res_json['imageHeight'])
            imageWidth = int(res_json['imageWidth'])
            for shape in shapes:
                points = shape['points']
                arr = []
                for point in points:
                    point_x = float(point[0])/imageWidth
                    point_y = float(point[1])/imageHeight
                    arr.append(str(point_x))
                    arr.append(str(point_y))
                out.write('0' + ' ' + ' '.join(arr) + '\n')

            f.close()
            out.close()


# 色差处理
def color_handle():
    path = r'E:\work\data\tower_crane\train_val_test'
    for data_type in ['train', 'val', 'test']:
        path1 = path + '/images/' + data_type
        path2 = path + '/labels/' + data_type

        img_names = os.listdir(path1)
        img_suff = '.png'

        index = 0
        print(len(img_names))
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
    path = r'E:\work\data\tower_crane\train_val_test'
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


def demo():
    image_source = r'E:\work\data\tower_crane\out\images\x01_1100-1-20191124.png'
    img = cv2.imread(image_source)
    width = img.shape[1]
    height = img.shape[0]
    print(width, height)
    # x01_1100-1-20211219_contour


if __name__ == '__main__':
    gaussian_salt()
