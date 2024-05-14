# coding=utf8
import os
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# 处理包含车子数据，主要适应openmmlab中模型
# data 3-dataset FAIR1M2.0
# data 3-dataset DOTA-v1.0
# data1 xview  dota-all/dota-xview
# data2 home forest\DOTA-v2.0
# data2 home forest\DOTA-BB
# data2 home forest\DOTA-tiny
# data2 home openmm\data\dior 待定


# xview  dota-all/dota-xview，E盘中有处理过的数据
def filter_and_copy_xview_data():
    path = r'E:\work\data\xview\dota-all\dota-xview\xview_split\data_with_original_label_format'
    label_out_path = r'E:\work\data\car20230705\xview\labels'
    image_out_path = r'E:\work\data\car20230705\xview\images'
    names = os.listdir(path)
    image_names = []
    label_names = []
    car_names = ['Small-Car', 'Truck', 'Cargo-Truck', 'Truck-w/Flatbed', 'Bus', 'Passenger-Vehicle', 'Utility-Truck',
                 'Dump-Truck', 'Crane-Truck', 'Truck-w/Box', 'Pickup-Truck', 'Cement-Mixer', 'Engineering-Vehicle',
                 'Passenger-Car', 'Cargo-Car', 'Locomotive', 'Tank-car', 'Flat-Car', 'Railway-Vehicle', 'Haul-Truck',
                 'Truck-w/Liquid']
    car_names1 = ['Small-Car', 'Utility-Truck']
    car_names2 = ['Truck', 'Cargo-Truck', 'Truck-w/Flatbed', 'Bus', 'Passenger-Vehicle',
                  'Dump-Truck', 'Crane-Truck', 'Truck-w/Box', 'Pickup-Truck', 'Cement-Mixer', 'Engineering-Vehicle',
                  'Haul-Truck',
                  'Truck-w/Liquid']
    for name in names:
        if name.endswith('.txt'):
            label_names.append(name)
        elif name.endswith('.png'):
            image_names.append(name)
    print('label_names,', len(label_names))
    for label_name in tqdm(label_names):
        f = open(path + '/' + label_name, encoding='utf8')
        res_arr = []
        for line in f:
            arr = line.replace('\n', '').split(' ')
            # 处理标签
            arr[0:8] = [float(x) for x in arr[0:8]]
            arr[0:8] = [arr[0], arr[1], arr[2] - arr[0], arr[3], arr[4] - arr[0], arr[5] - arr[1], arr[6],
                        arr[7] - arr[1]]
            arr = [str(x) for x in arr]

            label = arr[-1]
            location = arr[0:8]
            if label not in car_names1 and label not in car_names2:
                continue
            if label in car_names1:
                label = '0'
            else:
                label = '1'
            location.append(label)
            res_arr.append(location)
        f.close()

        if len(res_arr) > 0:
            out = open(label_out_path + '/' + label_name, 'w', encoding='utf8')
            for res in res_arr:
                out.write(' '.join(res) + '\n')
            out.close()
            """
            image_name = label_name.split('.')[0] + '.png'
            image_source_path = path + '/' + image_name
            image_target_path = image_out_path + '/' + image_name
            shutil.copy(image_source_path, image_target_path)
            """


# forest\DOTA-tiny
def copy_car_data_tiny():
    path = r'F:\data2\home\forest\DOTA-tiny'
    for file_name in ['test1024', 'train1024', 'val1024']:
        image_path = path + '/' + file_name + '/images'
        label_path = path + '/' + file_name + '/labelTxt'
        label_out_path = r'E:\work\data\car20230705\data_tiny\labels'
        image_out_path = r'E:\work\data\car20230705\data_tiny\images'
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        index = 0
        for label_name in label_names:
            index += 1
            if index % 1000 == 0:
                print(index)
            f = open(label_path + '/' + label_name, encoding='utf8')
            res_arr = []
            for line in f:
                arr = line.replace('\n', '').split(' ')
                label = arr[-2].lower()
                location = arr[0:8]

                if label == 'small-vehicle':
                    label = '0'
                elif label == 'large-vehicle':
                    label = '1'
                else:
                    continue
                location.append(label)
                res_arr.append(location)
            f.close()

            if len(res_arr) > 0:
                out = open(label_out_path + '/' + label_name, 'w', encoding='utf8')
                for res in res_arr:
                    out.write(' '.join(res) + '\n')
                out.close()
                image_name = label_name.split('.')[0] + '.png'
                image_source_path = image_path + '/' + image_name
                image_target_path = image_out_path + '/' + image_name
                shutil.copy(image_source_path, image_target_path)


# forest\DOTA-BB
def copy_car_data_bb():
    path = r'F:\data2\home\forest\DOTA-BB'
    image_path = path + '/images'
    label_path = path + '/labelTxt'
    label_out_path = r'E:\work\data\car20230705\data_bb\labels'
    image_out_path = r'E:\work\data\car20230705\data_bb\images'
    image_names = os.listdir(image_path)
    label_names = os.listdir(label_path)
    index = 0
    for label_name in label_names:
        index += 1
        if index % 1000 == 0:
            print(index)
        f = open(label_path + '/' + label_name, encoding='utf8')
        res_arr = []
        for line in f:
            arr = line.replace('\n', '').split(' ')
            label = arr[-2].lower()
            location = arr[0:8]

            if label == 'small-vehicle':
                label = '0'
            elif label == 'large-vehicle':
                label = '1'
            else:
                continue
            location.append(label)
            res_arr.append(location)
        f.close()

        if len(res_arr) > 0:
            out = open(label_out_path + '/' + label_name, 'w', encoding='utf8')
            for res in res_arr:
                out.write(' '.join(res) + '\n')
            out.close()
            image_name = label_name.split('.')[0] + '.png'
            image_source_path = image_path + '/' + image_name
            image_target_path = image_out_path + '/' + image_name
            shutil.copy(image_source_path, image_target_path)


# data 3-dataset FAIR1M2.0
def copy_car_fair1m2():
    path = r'E:\work\data\car20230705\FAIR1M2.0\source'
    image_out_path = r'E:\work\data\car20230705\FAIR1M2.0\images'
    label_out_path = r'E:\work\data\car20230705\FAIR1M2.0\labels'
    for file_name in ['train', 'validation']:
        image_path = path + '/' + file_name + '/images'
        label_path = path + '/' + file_name + '/labelXml'
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        index = 0
        for label_name in label_names:
            index += 1
            if index % 100 == 0:
                print(index)
            xmlname = label_path + '/' + label_name
            classes = ['Cargo Truck', 'Dump Truck', 'Small Car', 'Trailer', 'Truck Tractor', 'Van']
            in_file = open(xmlname, 'r')
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            objects = root.find('objects')
            res_arr = []
            for obj in objects.findall('object'):
                possibleresult = obj.find('possibleresult')
                cls = possibleresult.find('name').text
                points = obj.find('points')
                point_all = points.findall('point')
                location = []
                for point in point_all[0:-1]:
                    location.append(point.text.split(',')[0])
                    location.append(point.text.split(',')[1])
                if cls not in classes:
                    continue
                if cls == 'Small Car':
                    label = '0'
                else:
                    label = '1'
                location.append(label)
                res_arr.append(location)

            in_file.close()

            if len(res_arr) > 0:
                out = open(label_out_path + '/fair1m2_' + file_name + '_' + label_name.replace('.xml', '.txt'), 'w', encoding='utf8')
                for res in res_arr:
                    out.write(' '.join(res) + '\n')
                out.close()
                image_name = label_name.split('.')[0] + '.tif'
                image_source_path = image_path + '/' + image_name
                image_target_path = image_out_path + '/fair1m2_' + file_name + '_' + image_name.replace('.tif', '.png')
                shutil.copy(image_source_path, image_target_path)


# 处理data1.0车子
def copy_car_data1():
    path = r'E:\work\data\car20230705\data1.0\DOTA-v1.0'
    label_out_path = r'E:\work\data\car20230705\data1.0\labels'
    image_out_path = r'E:\work\data\car20230705\data1.0\images'
    for file_name in ['train', 'val']:
        image_path = path + '/' + file_name + '/images'
        label_path = path + '/' + file_name + '/labelTxt-v1.5/DOTA-v1.5_' + file_name
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        index = 0
        for label_name in label_names:
            index += 1
            if index % 100 == 0:
                print(index)
            f = open(label_path + '/' + label_name, encoding='utf8')
            res_arr = []
            for line in f:
                arr = line.replace('\n', '').split(' ')
                if len(arr) < 9:
                    continue
                label = arr[-2].lower()
                location = arr[0:8]

                if label == 'small-vehicle':
                    label = '0'
                elif label == 'large-vehicle':
                    label = '1'
                else:
                    continue
                location.append(label)
                res_arr.append(location)
            f.close()

            if len(res_arr) > 0:
                out = open(label_out_path + '/' + label_name, 'w', encoding='utf8')
                for res in res_arr:
                    out.write(' '.join(res) + '\n')
                out.close()
                image_name = label_name.split('.')[0] + '.png'
                image_source_path = image_path + '/' + image_name
                image_target_path = image_out_path + '/' + image_name
                shutil.copy(image_source_path, image_target_path)


# 处理data2.0车子
def copy_car_data2():
    path = r'E:\work\data\car20230705\data2.0\DOTA-v2.0'
    label_out_path = r'E:\work\data\car20230705\data2.0\labels'
    image_out_path = r'E:\work\data\car20230705\data2.0\images'
    for file_name in ['train', 'val', 'test']:
        image_path = path + '/' + file_name + '/images'
        label_path = path + '/' + file_name + '/labelTxt'
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        index = 0
        for label_name in label_names:
            index += 1
            if index % 100 == 0:
                print(index)
            f = open(label_path + '/' + label_name, encoding='utf8')
            res_arr = []
            for line in f:
                arr = line.replace('\n', '').split(' ')
                if len(arr) < 9:
                    continue
                label = arr[-2].lower()
                location = arr[0:8]

                if label == 'small-vehicle':
                    label = '0'
                elif label == 'large-vehicle':
                    label = '1'
                else:
                    continue
                location.append(label)
                res_arr.append(location)
            f.close()

            if len(res_arr) > 0:
                out = open(label_out_path + '/' + label_name, 'w', encoding='utf8')
                for res in res_arr:
                    out.write(' '.join(res) + '\n')
                out.close()
                image_name = label_name.split('.')[0] + '.png'
                image_source_path = image_path + '/' + image_name
                image_target_path = image_out_path + '/' + image_name
                shutil.copy(image_source_path, image_target_path)


# 查看车子图
def label2pic():
    # 读取图片
    image_path = r"E:\work\data\car20230705\xview\images"
    label_path = r'E:\work\data\car20230705\xview\labels'

    image_names = os.listdir(image_path)
    label_names = os.listdir(label_path)
    # 小型车、卡车、货运卡车、卡车-带/平板、总线、客运车辆、多功能卡车、自卸车、起重机-卡车、卡车-w/箱、皮卡车、水泥搅拌机、工程-车辆、客车、载货车、机车、坦克车、平板车、铁路车辆、运输卡车、卡车-带/液体
    car_names = ['Small-Car', 'Truck', 'Cargo-Truck', 'Truck-w/Flatbed', 'Bus', 'Passenger-Vehicle', 'Utility-Truck',
                 'Dump-Truck', 'Crane-Truck', 'Truck-w/Box', 'Pickup-Truck', 'Cement-Mixer', 'Engineering-Vehicle',
                 'Passenger-Car', 'Cargo-Car', 'Locomotive', 'Tank-car', 'Flat-Car', 'Railway-Vehicle', 'Haul-Truck',
                 'Truck-w/Liquid']
    """
    # 小车和大车
    car_names1 = ['Small-Car', 'Utility-Truck']
    car_names2 = ['Truck', 'Cargo-Truck', 'Truck-w/Flatbed', 'Bus', 'Passenger-Vehicle',
                 'Dump-Truck', 'Crane-Truck', 'Truck-w/Box', 'Pickup-Truck', 'Cement-Mixer', 'Engineering-Vehicle',
                 'Haul-Truck',
                 'Truck-w/Liquid']
    """
    for car_name in car_names:
        images_show_path = r'E:\work\data\car20230705\xview\show' + '\\' + car_name.replace('/', '_')
        os.mkdir(images_show_path)
        print(car_name)
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
            for line in f:
                arr = line.replace('\n', '').split(' ')
                location = []
                label = arr[-1]
                location.append([int(arr[0].split('.')[0]), int(arr[1].split('.')[0])])
                location.append([int(arr[2].split('.')[0]), int(arr[3].split('.')[0])])
                location.append([int(arr[4].split('.')[0]), int(arr[5].split('.')[0])])
                location.append([int(arr[6].split('.')[0]), int(arr[7].split('.')[0])])
                tmp_arr = np.array(location)
                if label == car_name:
                    contours.append(tmp_arr)

            if len(contours) > 0:
                img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
                cv2.imwrite(os.path.join(images_show_path, image_name.replace('.png', "_contour" + '.png')), img_contour)


if __name__ == '__main__':
    filter_and_copy_xview_data()
