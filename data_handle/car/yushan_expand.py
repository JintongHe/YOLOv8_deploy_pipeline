import json
import math
from osgeo import gdal
import os
import shutil

import numpy as np
import cv2
# 处理渔山岛0.8米数据


def get_new_points(points):
    # 四个点的经纬度（纬度，经度）
    # points = [[39.9042, 116.4074], [31.2304, 121.4737], [22.3964, 114.1095], [25.0330, 121.5654]]

    # 计算平均值
    avg_lat = sum([p[0] for p in points]) / len(points)
    avg_lon = sum([p[1] for p in points]) / len(points)
    print(avg_lat)

    # 计算每个点扩展一倍后的经纬度
    new_points = []
    for p in points:
        lat, lon = p
        # 计算距离
        distance = 6 * 6371.393 * math.asin(math.sqrt(math.sin((lat - avg_lat) / 2) ** 2 + math.cos(lat) * math.cos(avg_lat) * math.sin((lon - avg_lon) / 2) ** 2))
        # 计算新的经纬度
        new_lat = lat + (lat - avg_lat) / distance
        new_lon = lon + (lon - avg_lon) / distance
        new_points.append([new_lat, new_lon])

    # print(new_points)
    return new_points


# 处理小轿车
def handle_car():
    f = open('运输车.geojson', encoding='utf8')
    out = open('运输车_out_20230630.geojson', 'w', encoding='utf8')
    res = ''
    for line in f:
        res = res + line
    res_json = json.loads(res)
    print(res_json.keys())
    print(res_json['type'])
    print(res_json['crs'])
    print(res_json['features'][0])
    features = res_json['features']
    for feature in features:
        geometry = feature['geometry']
        coordinates = geometry['coordinates'][0]
        new_points = get_new_points(coordinates)
        print(coordinates, new_points)
        geometry['coordinates'][0] = new_points

    res_out = str(res_json).replace('\'', '"')
    out.write(res_out)

    f.close()
    out.close()


# 将坐标经纬度转换为像素位置
def mark_tif():
    # 读取图片信息
    # E:\work\python\ai\data_handle\car\03230303.tif
    image_path = r'E:\work\python\ai\data_handle\car\03230303.tif'
    in_ds = gdal.Open(image_path)  # 读取要切的原图
    print("open tif file succeed")
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    outbandsize = in_ds.RasterCount  # 获取数据波段数
    data = in_ds.ReadAsArray(0, 0, width, height)
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
    # 读取geojson信息
    f = open('小轿车_out_20230630.geojson', encoding='utf8')
    out = open('小轿车_xy_out_20230630.geojson', 'w', encoding='utf8')
    res = ''
    for line in f:
        res = res + line
    res_json = json.loads(res)
    print(res_json.keys())
    features = res_json['features']
    print(features[0])
    for feature in features:
        geometry = feature['geometry']
        coordinates = geometry['coordinates']
        for coordinate in coordinates:
            max_x = 0
            min_x = 1000000
            max_y = 0
            min_y = 1000000
            for lat_lon in coordinate:
                lat = lat_lon[0]
                lon = lat_lon[1]
                # 将geojson转换为像素坐标
                x = int((lat - adfGeoTransform[0])/adfGeoTransform[1])
                y = int((lon - adfGeoTransform[3])/adfGeoTransform[5])
                lat_lon[0] = x
                lat_lon[1] = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
            coordinate = [[min_x, max_y], [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            coordinates[0] = coordinate

    res_out = str(res_json).replace('\'', '"')
    out.write(res_out)

    f.close()
    out.close()


# 在图片中显示数据，查看标签像素点坐标是否正确
def label2pic():
    # 读取图片
    images_path = r"E:\work\python\ai\data_handle\car\03230303.tif"
    image_name = '03230303.tif'
    img = cv2.imread(images_path)

    w = img.shape[1]
    h = img.shape[0]
    # 读取标签
    # 读取geojson信息
    f = open('小轿车_xy_out_20230630.geojson', encoding='utf8')
    res = ''
    contours = []
    label_position = []
    for line in f:
        res = res + line
    res_json = json.loads(res)
    print(res_json.keys())
    features = res_json['features']
    print(features[0])
    for feature in features:
        geometry = feature['geometry']
        coordinates = geometry['coordinates']
        for coordinate in coordinates:
            tmp_arr = np.array(coordinate)
            contours.append(tmp_arr)
            label_position.append(tuple(coordinate[0]))

    img_contour = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    #for num in range(len(label_position)):
    #    img_contour = cv2.putText(img_contour, 'car', label_position[num], font, 0.4, color, 1)
    images_show_path = r'E:\work\python\ai\data_handle\car\test\show'
    cv2.imwrite(os.path.join(images_show_path, image_name.replace('.tif', "_contour" + '.tif')), img_contour)


# 将标签打到切割的图片上
def put_label_2_pic():
    # 读取geojson数据
    car_list1 = []
    car_list2 = []
    car_labels = ['小轿车_xy_out_20230630.geojson', '运输车_xy_out_20230630.geojson']
    for car_label in car_labels:
        f = open(r"E:\work\python\ai\data_handle\car" + '/' + car_label, encoding='utf8')
        res = ''
        for line in f:
            res = res + line
        res_json = json.loads(res)
        features = res_json['features']
        print(features[0])
        for feature in features:
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
            for coordinate in coordinates:
                if car_label == '小轿车_xy_out_20230630.geojson':
                    car_list1.append(coordinate[0:4])
                elif car_label == '运输车_xy_out_20230630.geojson':
                    car_list2.append(coordinate[0:4])
        f.close()

    # 读取图片信息
    image_path = r'E:\work\python\ai\data_handle\car\split'
    label_path = r'E:\work\python\ai\data_handle\car\label1'
    image_names = os.listdir(image_path)
    for image_name in image_names:
        arr = image_name.split('.')[0].split('_')
        # 当前图第一个像素点在原大图中的坐标
        base_name = arr[0] + '_1_1.tif'
        in_ds = gdal.Open(image_path + '/' + base_name)
        width = in_ds.RasterXSize  # 获取数据宽度
        height = in_ds.RasterYSize  # 获取数据高度
        x0 = width * (int(arr[1]) - 1)
        y0 = height * (int(arr[2]) - 1)
        # 最后一个像素点在大图中的坐标
        in_ds1 = gdal.Open(image_path + '/' + image_name)
        width1 = in_ds1.RasterXSize  # 获取数据宽度
        height1 = in_ds1.RasterYSize  # 获取数据高度
        x1 = x0 + width1 - 1
        y1 = y0 + height1 - 1
        car1_res = []
        car2_res = []
        for car_list in [car_list1, car_list2]:
            for car_location in car_list:
                max_x = 0
                min_x = 1000000
                max_y = 0
                min_y = 1000000
                for location in car_location:
                    x = location[0]
                    y = location[1]
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                if min_x >= x0 and max_x <= x1 and min_y >= y0 and max_y <= y1:
                    if len(car_list) == len(car_list1):
                        car1_res.append(car_location)
                    elif len(car_list) == len(car_list2):
                        car2_res.append(car_location)

        if len(car1_res) != 0 or len(car2_res) != 0:
            out = open(label_path + '/' + image_name.split('.')[0] + '.txt', 'w', encoding='utf8')
            for car_location in car1_res:
                res_location = []
                for location in car_location:
                    res_location.append(str(location[0] - x0))
                    res_location.append(str(location[1] - y0))
                out.write(' '.join(res_location) + ' ' + '0' + '\n')
            for car_location in car2_res:
                res_location = []
                for location in car_location:
                    res_location.append(str(location[0] - x0))
                    res_location.append(str(location[1] - y0))
                out.write(' '.join(res_location) + ' ' + '1' + '\n')
            out.close()


# 在分割图片中显示数据，查看标签像素点坐标是否正确
def split_label2pic():
    # 读取图片
    images_path = r"E:\work\python\ai\data_handle\car\split"
    label_path = r'E:\work\python\ai\data_handle\car\label1'
    image_names = os.listdir(images_path)
    label_names = os.listdir(label_path)
    index = 0
    for image_name in image_names:
        index += 1
        if index % 50 == 0:
            print(index)
        label_name = image_name.split('.')[0] + '.txt'
        if label_name not in label_names:
            continue
        img = cv2.imread(images_path + '/' + image_name)
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
            location.append([int(arr[0]), int(arr[1])])
            location.append([int(arr[2]), int(arr[3])])
            location.append([int(arr[4]), int(arr[5])])
            location.append([int(arr[6]), int(arr[7])])
            tmp_arr = np.array(location)
            if label == '0':
                contours.append(tmp_arr)
            elif label == '1':
                contours1.append(tmp_arr)

        img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
        img_contour = cv2.drawContours(img_contour, contours1, -1, (255, 255, 255), 1)

        images_show_path = r'E:\work\python\ai\data_handle\car\split_show'
        cv2.imwrite(os.path.join(images_show_path, image_name.replace('.tif', "_contour" + '.tif')), img_contour)
        break


# 将数据存在data目录下
def copy_data():
    label_path = r'E:\work\python\ai\data_handle\car\label1'
    image_path = r'E:\work\python\ai\data_handle\car\split'
    label_names = os.listdir(label_path)
    image_names = os.listdir(image_path)
    label_out_path = r'E:\work\data\car20230705\yushan\labels'
    image_out_path = r'E:\work\data\car20230705\yushan\images'
    for label_name in label_names:
        label_source_path = label_path + '/' + label_name
        label_target_path = label_out_path + '/' + label_name
        shutil.copy(label_source_path, label_target_path)
        image_source_path = image_path + '/' + label_name.split('.')[0] + '.tif'
        image_target_path = image_out_path + '/' + label_name.split('.')[0] + '.png'
        shutil.copy(image_source_path, image_target_path)


if __name__ == '__main__':
    split_label2pic()
