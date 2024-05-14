# coding=utf8
import os
import shutil
from shutil import copyfile
from tqdm import tqdm
from osgeo import gdal
import random
import numpy as np
import cv2


def merge_all_images():
    path = r'E:\work\data\car20230705'
    # file_names = ['data_tiny', 'data1.0', 'data2.0', 'FAIR1M2.0', 'xview', 'yushan']
    file_names = ['data_tiny', 'data1.0', 'data2.0', 'xview', 'yushan']
    image_out_path = path + '/all/20230720/images'
    label_out_path = path + '/all/20230720/labels'
    sets = set()
    same_num = 0
    for file_name in file_names:
        print(file_name)
        image_path = path + '/' + file_name + '/images'
        label_path = path + '/' + file_name + '/labels'
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        for label_name in tqdm(label_names):
            if label_name in sets:
                print('数据重复:', file_name, label_name)
                same_num += 1
                continue
            else:
                sets.add(label_name)
            image_name = label_name.replace('.txt', '.png')
            image_source_path = image_path + '/' + image_name
            image_target_path = image_out_path + '/' + image_name
            label_source_path = label_path + '/' + label_name
            label_target_path = label_out_path + '/' + label_name
            shutil.copy(image_source_path, image_target_path)
            shutil.copy(label_source_path, label_target_path)

    print('重复数据量:', same_num)


# 处理xview的标签数据
def handle_xview_label():
    xview_label_path = r'E:\work\data\car20230705\xview\labels'
    xview_labels = os.listdir(xview_label_path)
    label_path = r'E:\work\data\car20230705\all\labels'
    for xview_label in tqdm(xview_labels):
        f = open(xview_label_path + '/' + xview_label, encoding='utf8')
        out1 = open(label_path + '/' + xview_label, 'w', encoding='utf8')
        out2 = open(label_path + '/' + xview_label.replace('.txt', '_r1_.txt'), 'w', encoding='utf8')
        for line in f:
            arr = line.replace('\n', '').split(' ')
            arr[0:8] = [float(x) for x in arr]
            arr[0:8] = [arr[0], arr[1], arr[2]-arr[0], arr[3], arr[4]-arr[0], arr[5]-arr[1], arr[6], arr[7]-arr[1]]
            arr = [str(x) for x in arr]
            out1.write(' '.join(arr) + '\n')
            out2.write(' '.join(arr) + '\n')
        f.close()
        out1.close()
        out2.close()


# 查找大图像
def find_large_images():
    image_path = r'E:\work\data\car20230705\all\images'
    label_path = r'E:\work\data\car20230705\all\labels'
    out_path = r'E:\work\data\car20230705\all'
    image_names = os.listdir(image_path)
    large_image_num = 0
    for image_name in tqdm(image_names):
        label_name = image_name.replace('.png', '.txt')
        image_source_path = image_path + '/' + image_name
        label_source_path = label_path + '/' + label_name
        in_ds = gdal.Open(image_source_path)
        width = in_ds.RasterXSize  # 获取数据宽度
        height = in_ds.RasterYSize  # 获取数据高度
        if width >= 2000 and height >= 2000:
            large_image_num += 1
            print(width, height)
            image_target_path = out_path + '/images_large/' + image_name
            shutil.copy(image_source_path, image_target_path)

            label_target_path = out_path + '/labels_large/' + label_name
            shutil.copy(label_source_path, label_target_path)

        else:
            image_target_path = out_path + '/images_small/' + image_name
            shutil.copy(image_source_path, image_target_path)

            label_target_path = out_path + '/labels_small/' + label_name
            shutil.copy(label_source_path, label_target_path)

    print(large_image_num)


# 色差处理
def color_handle():
    image_path = r'E:\work\data\car20230705\all\20230720\images'
    label_path = r'E:\work\data\car20230705\all\20230720\labels'
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
        # ---------------------------------
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] * 0.5
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
        img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(file1.replace(img_suff, "_r2_" + img_suff), img1)
        shutil.copy(file2, file2.replace(".txt", "_r2_.txt"))
        """


# 在分割图片中显示数据，查看标签像素点坐标是否正确
def label2pic():
    # 读取图片
    # image_path = r"E:\work\data\car20230705\all\images"
    # label_path = r'E:\work\data\car20230705\all\labels'
    # images_show_path = r'E:\work\data\car20230705\all\show'
    image_path = r"E:\work\data\car20230705\xview\images"
    label_path = r'E:\work\data\car20230705\xview\labels'
    images_show_path = r'E:\work\data\car20230705\xview\show'
    images_show_path1 = r'E:\work\data\car20230705\xview\show1'
    image_names = os.listdir(image_path)
    label_names = os.listdir(label_path)
    all = [0, 0]
    for image_name in tqdm(image_names):
        label_name = image_name.split('.')[0] + '.txt'
        if label_name not in label_names:
            continue
        img = cv2.imread(image_path + '/' + image_name)
        w = img.shape[1]
        h = img.shape[0]
        # 读取标签
        f = open(label_path + '/' + label_name, encoding='utf8')
        contours = []
        contours1 = []
        for line in f:
            arr = line.replace('\n', '').split(' ')
            location = []
            label = arr[-1]
            location.append([int(arr[0].split('.')[0]), int(arr[1].split('.')[0])])
            location.append([int(arr[2].split('.')[0]), int(arr[3].split('.')[0])])
            location.append([int(arr[4].split('.')[0]), int(arr[5].split('.')[0])])
            location.append([int(arr[6].split('.')[0]), int(arr[7].split('.')[0])])
            tmp_arr = np.array(location)
            if label == '0':
                contours.append(tmp_arr)
            elif label == '1':
                contours1.append(tmp_arr)

        img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
        img_contour = cv2.drawContours(img_contour, contours1, -1, (255, 255, 255), 1)
        all[0] += len(contours)
        all[1] += len(contours1)
        '''
        if len(contours) + len(contours1) > 20:
            cv2.imwrite(os.path.join(images_show_path, image_name.replace('.png', "_contour" + '.png')), img_contour)
            shutil.copy(image_path + '/' + image_name, images_show_path + '/' + image_name)
        else:
            cv2.imwrite(os.path.join(images_show_path1, image_name.replace('.png', "_contour" + '.png')), img_contour)
            shutil.copy(image_path + '/' + image_name, images_show_path1 + '/' + image_name)
        '''

    print('all,', all)


# 生成train、test、val数据
def get_train_test_val():
    path = r'E:\work\data\car20230705\all'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    target_path = r'E:\work\data\car20230705\all\train_val_test'
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
        copyfile(image_source, image_target)
        copyfile(label_source, label_target)


# 将标签转换为json格式


if __name__ == '__main__':
    label2pic()
