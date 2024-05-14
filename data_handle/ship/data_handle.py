# coding = utf8
import json
import os
from tqdm import tqdm
from shutil import copyfile
import shutil
import cv2
import numpy as np
import gdal


def data_clean():
    source_path = r'E:\work\data\ship\source'
    source1_path = r'E:\work\data\ship\source1'
    # 数据从data，data1，data2，data3中找的，手动删除或移动了一些数据，去重或者使得数据更好处理
    source_file_names = os.listdir(source_path)
    for source_file_name in source_file_names:
        print(source_file_name)
        if source_file_name == 'coco_aitod':
            # ship 目标检测的实例分割
            '''
            f = open(source_path + '/' + source_file_name + '/annotations/instances_testv1.json', encoding='utf8')
            res = ''
            tem_index = 0
            for line in tqdm(f):
                tem_index += 1
                if tem_index % 500 == 0:
                    print(tem_index)
                res = res + line
            f.close()
            json_res = json.loads(res)
            annotations = json_res['annotations']
            images = json_res['images']
            image_id = 0
            for image in tqdm(images):
                image_id += 1
                file_name = image['file_name']
                segmentations = []
                for annotation in annotations:
                    category_id = annotation['category_id']
                    if category_id != 4:
                        continue
                    image_id1 = annotation['image_id']
                    if image_id1 != image_id:
                        continue
                    segmentation = annotation['segmentation'][0]
                    segmentations.append(segmentation)
                if len(segmentations) != 0:
                    out = open(source1_path + '/' + source_file_name + '/labels/' + file_name.split('.')[0] + '.txt', 'w', encoding='utf8')
                    for segmentation in segmentations:
                        out.write(' '.join(segmentation) + ' 0\n')
                    out.close()
            '''
            continue
        if source_file_name == 'data':
            # 'ship'
            # 'dior' 目标检测的实例分割
            # xview_split' 目标检测的实例分割
            pass
        if source_file_name == 'DOTA-BB':
            # 'ship'
            # 目标检测的实例分割
            pass
        if source_file_name == 'DOTA-tiny':
            # 'ship'
            '''
            file_names = os.listdir(source_path + '/' + source_file_name)
            print(file_names)
            for file_name in file_names:
                label_path = source_path + '/' + source_file_name + '/' + file_name + '/labelTxt'
                image_path = source_path + '/' + source_file_name + '/' + file_name + '/images'
                image_names = os.listdir(image_path)
                label_names = os.listdir(label_path)
                for label_name in tqdm(label_names):
                    f = open(label_path + '/' + label_name, encoding='utf8')
                    res = []
                    for line in f:
                        arr = line.replace('\n', '').split(' ')
                        label = arr[-2]
                        if label != 'ship':
                            continue
                        res.append(arr[0:8])
                    f.close()
                    if len(res) == 0:
                        continue
                    out = open(source1_path + '/' + source_file_name + '/labels/' + label_name, 'w', encoding='utf8')
                    for arr in res:
                        out.write(' '.join(arr) + '\n')
                    out.close()
                    image_name = label_name.split('.')[0] + '.png'
                    image_source_path = image_path + '/' + label_name.split('.')[0] + '.png'
                    image_target_path = source1_path + '/' + source_file_name + '/images/' + image_name
                    shutil.copy(image_source_path, image_target_path)
            '''

        if source_file_name == 'DOTA-v1.0':
            # 'ship'
            '''
            file_names = os.listdir(source_path + '/' + source_file_name)
            print(file_names)
            for file_name in file_names:
                label_path = source_path + '/' + source_file_name + '/' + file_name + '/DOTA-v1.5'
                image_path = source_path + '/' + source_file_name + '/' + file_name + '/images'
                image_names = os.listdir(image_path)
                label_names = os.listdir(label_path)
                for label_name in tqdm(label_names):
                    f = open(label_path + '/' + label_name, encoding='utf8')
                    res = []
                    for line in f:
                        arr = line.replace('\n', '').split(' ')
                        if len(arr) < 8:
                            continue
                        label = arr[-2]
                        if label != 'ship':
                            continue
                        res.append(arr[0:8])
                    f.close()
                    if len(res) == 0:
                        continue
                    out = open(source1_path + '/' + source_file_name + '/labels/' + label_name, 'w', encoding='utf8')
                    for arr in res:
                        out.write(' '.join(arr) + '\n')
                    out.close()
                    image_name = label_name.split('.')[0] + '.png'
                    image_source_path = image_path + '/' + label_name.split('.')[0] + '.png'
                    image_target_path = source1_path + '/' + source_file_name + '/images/' + image_name
                    shutil.copy(image_source_path, image_target_path)
            '''
        if source_file_name == 'DOTA-v2.0':
            '''
            # 'ship'
            file_names = os.listdir(source_path + '/' + source_file_name)
            print(file_names)
            for file_name in file_names:
                label_path = source_path + '/' + source_file_name + '/' + file_name + '/labelTxt'
                image_path = source_path + '/' + source_file_name + '/' + file_name + '/images'
                image_names = os.listdir(image_path)
                label_names = os.listdir(label_path)
                for label_name in tqdm(label_names):
                    if '.' not in label_name:
                        continue
                    f = open(label_path + '/' + label_name, encoding='utf8')
                    res = []
                    for line in f:
                        arr = line.replace('\n', '').split(' ')
                        if len(arr) < 8:
                            continue
                        label = arr[-2]
                        if label != 'ship':
                            continue
                        res.append(arr[0:8])
                    f.close()
                    if len(res) == 0:
                        continue
                    image_name = label_name.split('.')[0] + '.png'
                    if image_name not in image_names:
                        continue
                    out = open(source1_path + '/' + source_file_name + '/labels/' + label_name, 'w', encoding='utf8')
                    for arr in res:
                        out.write(' '.join(arr) + '\n')
                    out.close()
                    image_name = label_name.split('.')[0] + '.png'
                    image_source_path = image_path + '/' + label_name.split('.')[0] + '.png'
                    image_target_path = source1_path + '/' + source_file_name + '/images/' + image_name
                    shutil.copy(image_source_path, image_target_path)
            '''
        if source_file_name == 'dota-xview':
            # 目标检测的实例分割
            continue
        if source_file_name == '石化':
            file_names = os.listdir(source_path + '/' + source_file_name)
            for file_name in file_names:
                geojson_names = os.listdir(source_path + '/' + source_file_name + '/' + file_name + '/geojson')
                image_names = os.listdir(source_path + '/' + source_file_name + '/' + file_name + '/影像')
                image_name = image_names[0]
                in_ds = gdal.Open(source_path + '/' + source_file_name + '/' + file_name + '/影像/' + image_name)  # 读取要切的原图
                width = in_ds.RasterXSize  # 获取数据宽度
                height = in_ds.RasterYSize  # 获取数据高度
                outbandsize = in_ds.RasterCount  # 获取数据波段数
                print(width, height, outbandsize)
                # 读取图片地理信息
                adfGeoTransform = in_ds.GetGeoTransform()
                print(adfGeoTransform)
                print(adfGeoTransform[0])
                print(adfGeoTransform[3])
                print(adfGeoTransform[1])
                print(adfGeoTransform[2])
                print(adfGeoTransform[4])
                print(adfGeoTransform[5])
                res = []
                for geojson_name in geojson_names:
                    if '船' not in geojson_name:
                        continue
                    f = open(source_path + '/' + source_file_name + '/' + file_name + '/geojson/' + geojson_name, encoding='utf8')
                    tmp_label = ''
                    for line in f:
                        tmp_label = tmp_label + line
                    f.close()
                    label_json = json.loads(tmp_label)
                    features = label_json['features']
                    for feature in features:
                        geometry = feature['geometry']
                        coordinates = geometry['coordinates']
                        coordinate = coordinates[0]
                        coordinate_arr = []
                        for coordinate_index in range(len(coordinate)-1):
                            xy = coordinate[coordinate_index]
                            # 经纬度坐标转换为像素点坐标
                            lat = xy[0]
                            lon = xy[1]
                            x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                            y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                            coordinate_arr.append(x)
                            coordinate_arr.append(y)
                        if len(coordinate_arr) > 0:
                            res.append(coordinate_arr)

                label_name = image_name.split('.')[0] + '.txt'
                out = open(source1_path + '/mark_data/labels/' + label_name, 'w', encoding='utf8')
                for arr in res:
                    arr = [str(x) for x in arr]
                    out.write(' '.join(arr) + '\n')
                out.close()
                image_source_path = source_path + '/' + source_file_name + '/' + file_name + '/影像/' + image_name
                image_target_path = source1_path + '/mark_data/images/' + image_name
                shutil.copy(image_source_path, image_target_path)
            # 目标检测的实例分割
            continue


# 标签展示
def label2pic():
    path = r'E:\work\data\ship\source1\mark_data'

    image_names = os.listdir(path + '/images_split')
    label_names = os.listdir(path + '/labels_split')
    images_show_path = path + '/show_split'
    for image_name in tqdm(image_names):
        img = cv2.imread(path + '/images_split/' + image_name)

        label_name = image_name.split('.')[0] + '.txt'
        f = open(path + '/labels_split/' + label_name, encoding='utf8')
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
        f.close()
        # 标签打到图片上
        img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)

        cv2.imwrite(os.path.join(images_show_path, image_name.replace('.tif', "_contour" + '.tif')), img_contour)


# 合并数据
def merge_data():
    images_target = 'E:/work/data/ship/images'
    labels_target = 'E:/work/data/ship/labels'
    source1 = r'E:\work\data\ship\source1'
    file1_names = os.listdir(source1)
    for file1_name in file1_names:
        print(file1_name)
        image_names = os.listdir(source1 + '/' + file1_name + '/images')
        label_names = os.listdir(source1 + '/' + file1_name + '/labels')

        for image_name in tqdm(image_names):
            source_path = source1 + '/' + file1_name + '/images/' + image_name
            target_path = images_target + '/' + image_name
            copyfile(source_path, target_path)

        for label_name in tqdm(label_names):
            source_path = source1 + '/' + file1_name + '/labels/' + label_name
            target_path = labels_target + '/' + label_name
            copyfile(source_path, target_path)


# 转换为yolo格式的label
def label2yolo():
    label_path = 'E:/work/data/ship/source1/mark_data/labels_split'
    image_path = 'E:/work/data/ship/source1/mark_data/images_split'
    label_yolo_path = 'E:/work/data/ship/source1/mark_data/labels_yolo_split'
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


def split_image(image_path, split_arr):
    '''
    image_path = r'E:/work/data/ship/source1/mark_data/images/20230303.tif'
    split_arr = [[1300, 1300], [800, 800], [1000, 700], [700, 1000], [1600, 1600], [2500, 2500]]
    split_image(image_path, split_arr)
    '''
    print('开始切割图片')

    image_name = image_path.split('/')[-1]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    split_out_path = path + '/out/' + image_name.replace('.' + suf, '_split')
    if not os.path.exists(split_out_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(split_out_path)

    num = 0
    for split_data in split_arr:
        print(split_data)
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        picture = cv2.imread(path + '/' + image_name)
        (height, width, depth) = picture.shape
        # 预处理生成0矩阵
        pic = np.zeros((cut_height, cut_width, depth))
        # 计算可以划分的横纵的个数
        num_width = int((width + cut_width * 4 / 5) / cut_width)
        num_height = int((height + cut_height * 4 / 5) / cut_height)
        # for循环迭代生成
        for i in range(0, num_width):
            for j in range(0, num_height):
                pic = picture[j * cut_height: (j + 1) * cut_height, i * cut_width: (i + 1) * cut_width, :]
                result_path = split_out_path + '/split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                cv2.imwrite(result_path, pic)
                '''
                # 多重切割处理
                if i * cut_width + cut_width < width:
                    pic1 = picture[j * cut_height: (j + 1) * cut_height,
                           i * cut_width + int(cut_width / 2): (i + 1) * cut_width + int(cut_width / 2), :]
                    result_path1 = split_out_path + '/r1split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
                if j * cut_height + cut_height < height:
                    pic1 = picture[j * cut_height + int(cut_height / 2): (j + 1) * cut_height + int(cut_height / 2),
                           i * cut_width: (i + 1) * cut_width, :]
                    result_path1 = split_out_path + '/r2split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
                if i * cut_width + cut_width < width and j * cut_height + cut_height < height:
                    pic1 = picture[j * cut_height + int(cut_height / 2): (j + 1) * cut_height + int(cut_height / 2),
                           i * cut_width + int(cut_width / 2): (i + 1) * cut_width + int(cut_width / 2), :]
                    result_path1 = split_out_path + '/r3split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
                '''
        num += 1

    print("切割图片完成!!!")
    return split_out_path


# 将标签打到切割的图片上
def put_label_2_pic():
    split_arr = [[1300, 1300], [800, 800], [1000, 700], [700, 1000], [1600, 1600], [2500, 2500]]
    image_path = r'E:\work\data\ship\source1\mark_data\images'
    label_path = r'E:\work\data\ship\source1\mark_data\labels'
    split_out_label_path = r'E:\work\data\ship\source1\mark_data\labels_split'
    split_out_image_path = r'E:\work\data\ship\source1\mark_data\images_split'
    image_names = os.listdir(image_path)
    for image_name in image_names:
        if '.' not in image_name:
            continue
        label_name = image_name.split('.')[0] + '.txt'
        label_list = []
        f = open(label_path + '/' + label_name, encoding='utf8')
        for line in f:
            label_list.append(line.replace('\n', '').split(' '))
        f.close()
        split_path = image_path + '/out/' + image_name.replace('.tif', '_split')
        split_images = os.listdir(split_path)
        for split_image in split_images:
            arr = split_image.split('.')[0].split('_')
            split_type = int(arr[0].split('split')[1])
            width = split_arr[split_type][0]  # 获取数据宽度
            height = split_arr[split_type][1]  # 获取数据高度
            x0 = width * (int(arr[1]) - 1)
            y0 = height * (int(arr[2]) - 1)
            # 最后一个像素点在大图中的坐标
            in_ds1 = gdal.Open(split_path + '/' + split_image)
            width1 = in_ds1.RasterXSize  # 获取数据宽度
            height1 = in_ds1.RasterYSize  # 获取数据高度
            x1 = x0 + width1 - 1
            y1 = y0 + height1 - 1
            res_list = []
            for location in label_list:
                max_x = 0
                min_x = 1000000
                max_y = 0
                min_y = 1000000
                location_tmp = []
                for index in range(0, len(location), 2):
                    x = int(location[index])
                    y = int(location[index+1])
                    location_tmp.append(x - x0)
                    location_tmp.append(y - y0)
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                if min_x >= x0 and max_x <= x1 and min_y >= y0 and max_y <= y1:
                    location_tmp = [str(x) for x in location_tmp]
                    res_list.append(location_tmp)
            # 写入
            if len(res_list) != 0:
                out = open(split_out_label_path + '/' + split_image.split('.')[0] + '.txt', 'w', encoding='utf8')
                for location in res_list:
                    out.write(' '.join(location) + '\n')
                out.close()
                image_source_path = split_path + '/' + split_image
                image_target_path = split_out_image_path + '/' + split_image.replace('.tif', '.png')
                shutil.copy(image_source_path, image_target_path)


if __name__ == '__main__':
    label2yolo()
