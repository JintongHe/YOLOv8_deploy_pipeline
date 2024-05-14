# coding = utf8
import geopandas as gpd
import os
import zipfile
from osgeo import gdal
import json
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from shutil import copyfile
# 处理车子标注数据


# 转换为geojson文件
def batch_read_shapfile():
    source_dir = r'E:\work\data\car20230705\mark'
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        if file_name.endswith('.zip') or file_name == 'images' or file_name == 'labels' or file_name == 'show':
            continue
        subfile1_names = os.listdir(source_dir + '/' + file_name)
        # 将shapfile文件批量转为geojson文件
        for subfile1_name in subfile1_names:
            subfile2_names = os.listdir(source_dir + '/' + file_name + '/' + subfile1_name)
            for subfile2_name in subfile2_names:
                if '.' in subfile2_name:
                    continue
                subfile3_names = os.listdir(source_dir + '/' + file_name + '/' + subfile1_name + '/' + subfile2_name)
                for subfile3_name in subfile3_names:
                    if not subfile3_name.endswith('.shp'):
                        continue
                    target_dir = source_dir + '/' + file_name + '/' + subfile1_name
                    target_filename = os.path.basename(subfile3_name).split(".")[0]
                    data = gpd.read_file(source_dir + '/' + file_name + '/' + subfile1_name + '/' + subfile2_name + '/' + subfile3_name)
                    geojson_file = os.path.join(target_dir, target_filename + ".geojson")

                    data.crs = 'EPSG:4326'
                    data.to_file(geojson_file, driver="GeoJSON")
                    # gbk格式转换为utf-8格式
                    with open(geojson_file, 'r', encoding='gbk') as f:
                        content = f.read()
                    with open(geojson_file, 'w', encoding='utf8') as f:
                        f.write(content)

# 图像切割，见split_joint_predict.py中mark_split方法

# 将坐标经纬度转换为像素位置，用txt文件存储
def mark_tif():
    source_dir = r'E:/work/data/car20230705/mark'
    file1_names = os.listdir(source_dir)
    for file1_name in file1_names:
        if '.' in file1_name or file1_name == 'images' or file1_name == 'labels' or file1_name == 'show':
            continue
        file2_names = os.listdir(source_dir + '/' + file1_name)
        for file2_name in file2_names:
            file3_names = os.listdir(source_dir + '/' + file1_name + '/' + file2_name)
            file3_label_names = []
            # 获取geojson数据
            for file3_name in file3_names:
                if '.geojson' not in file3_name:
                    continue
                if '小轿车' in file3_name or '运输车' in file3_name:
                    file3_label_names.append(file3_name)
            # 获取tif文件，生成标签数据
            for file3_name in file3_names:
                file3_name_replace = file3_name.replace(' ', '').replace('（', '(').replace('）', ')')
                if '(18).tif' in file3_name_replace:
                    continue
                if not file3_name.endswith('.tif'):
                    continue
                label_name = file3_name.replace('.tif', '.txt')
                label_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + label_name
                out = open(label_path, 'w', encoding='utf8')
                for file3_label_name in file3_label_names:
                    label = 0
                    if '小轿车' in file3_label_name:
                        label = 0
                    elif '运输车' in file3_label_name:
                        label = 1
                    else:
                        print('error, 出错啦，检查label')
                    # 解析json
                    image_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_name

                    geojson_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_label_name
                    in_ds = gdal.Open(image_path)  # 读取要切的原图
                    print("open tif file succeed")
                    print(geojson_path)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    outbandsize = in_ds.RasterCount  # 获取数据波段数
                    print(width, height, outbandsize)
                    # 读取图片地理信息
                    adfGeoTransform = in_ds.GetGeoTransform()
                    print(adfGeoTransform)
                    print('\n')
                    # 读取geojson信息
                    f = open(geojson_path, encoding='utf8')

                    res = ''
                    for line in f:
                        res = res + line
                    res_json = json.loads(res)
                    features = res_json['features']

                    for feature in features:
                        geometry = feature['geometry']
                        coordinates = geometry['coordinates']
                        label_res = ''
                        for coordinate in coordinates:
                            for lat_lon in coordinate[0:-1]:
                                lat = lat_lon[0]
                                lon = lat_lon[1]
                                # 将geojson转换为像素坐标
                                x = int((lat - adfGeoTransform[0]) / adfGeoTransform[1])
                                y = int((lon - adfGeoTransform[3]) / adfGeoTransform[5])
                                label_res = label_res + ' ' + str(x) + ' ' + str(y)

                        if label_res != '':
                            label_res = str(label) + label_res
                            out.write(label_res + '\n')

                    f.close()

                out.close()


# 将标签打到切割的图片上
def put_label_2_pic():
    source_dir = r'E:/work/data/car20230705/mark'
    file1_names = os.listdir(source_dir)
    for file1_name in file1_names:
        if '.' in file1_name or file1_name == 'images' or file1_name == 'labels' or file1_name == 'show':
            continue
        file2_names = os.listdir(source_dir + '/' + file1_name)
        for file2_name in file2_names:
            file3_names = os.listdir(source_dir + '/' + file1_name + '/' + file2_name)
            file3_label_names = []
            # 获取txt数据
            for file3_name in file3_names:
                if not file3_name.endswith('.txt'):
                    continue
                file3_label_names.append(file3_name)
            for file3_label_name in file3_label_names:
                print(file2_name, file3_label_name)
                car_list = []
                label_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_label_name
                f = open(label_path, encoding='utf8')
                for line in f:
                    car_list.append(line.replace('\n', '').split(' '))
                # 读取split数据
                split_name = file3_label_name.replace('.txt', 'split')
                split_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + split_name
                split_label_name = split_name + '_label'
                split_label_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + split_label_name
                if not os.path.exists(split_label_path):
                    os.makedirs(split_label_path)
                split_images = os.listdir(split_path)
                for split_image in split_images:
                    arr = split_image.split('.')[0].split('_')
                    base_name = arr[0] + '_1_1.tif'
                    in_ds = gdal.Open(split_path + '/' + base_name)
                    width = in_ds.RasterXSize  # 获取数据宽度
                    height = in_ds.RasterYSize  # 获取数据高度
                    x0 = width * (int(arr[1]) - 1)
                    y0 = height * (int(arr[2]) - 1)
                    # 最后一个像素点在大图中的坐标
                    in_ds1 = gdal.Open(split_path + '/' + split_image)
                    width1 = in_ds1.RasterXSize  # 获取数据宽度
                    height1 = in_ds1.RasterYSize  # 获取数据高度
                    x1 = x0 + width1 - 1
                    y1 = y0 + height1 - 1
                    car_res = []
                    for car_location in car_list:
                        max_x = 0
                        min_x = 1000000
                        max_y = 0
                        min_y = 1000000
                        car_location_tmp = [car_location[0]]
                        for index in range(1, len(car_location), 2):
                            x = int(car_location[index])
                            y = int(car_location[index+1])
                            car_location_tmp.append(x - x0)
                            car_location_tmp.append(y - y0)
                            if x > max_x:
                                max_x = x
                            if y > max_y:
                                max_y = y
                            if x < min_x:
                                min_x = x
                            if y < min_y:
                                min_y = y
                        if min_x >= x0 and max_x <= x1 and min_y >= y0 and max_y <= y1:
                            car_location_tmp = [str(x) for x in car_location_tmp]
                            car_res.append(car_location_tmp)
                    # 写入
                    if len(car_res) != 0:
                        out = open(split_label_path + '/' + split_image.split('.')[0] + '.txt', 'w', encoding='utf8')
                        for car_location in car_res:
                            out.write(' '.join(car_location) + '\n')
                        out.close()


# 在分割图片中显示数据，查看标签像素点坐标是否正确
def split_label2pic():
    source_dir = r'E:/work/data/car20230705/mark'
    file1_names = os.listdir(source_dir)
    for file1_name in file1_names:
        if '.' in file1_name or file1_name == 'images' or file1_name == 'labels' or file1_name == 'show':
            continue
        file2_names = os.listdir(source_dir + '/' + file1_name)
        for file2_name in file2_names:
            file3_names = os.listdir(source_dir + '/' + file1_name + '/' + file2_name)
            file3_label_names = []
            for file3_name in file3_names:
                if not file3_name.endswith('_label'):
                    continue
                file3_label_names.append(file3_name)
            for file3_label_name in file3_label_names:
                print(file2_name, file3_label_name)
                image_file_name = file3_label_name.replace('_label', '')
                image_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + image_file_name
                label_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_label_name
                show_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_label_name.replace('_label', '_show')
                if not os.path.exists(show_path):
                    os.makedirs(show_path)

                # 读取图片
                image_names = os.listdir(image_path)
                label_names = os.listdir(label_path)
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
                        label = arr[0]
                        location.append([int(arr[1]), int(arr[2])])
                        location.append([int(arr[3]), int(arr[4])])
                        location.append([int(arr[5]), int(arr[6])])
                        location.append([int(arr[7]), int(arr[8])])
                        tmp_arr = np.array(location)
                        if label == '0':
                            contours.append(tmp_arr)
                        elif label == '1':
                            contours1.append(tmp_arr)

                    img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
                    img_contour = cv2.drawContours(img_contour, contours1, -1, (255, 255, 255), 1)

                    cv2.imwrite(os.path.join(show_path, image_name.replace('.tif', "_contour" + '.tif')), img_contour)


# 合并数据
def merge_images_labels():
    index = 1
    image_path = r'E:\work\data\car20230705\mark\images'
    label_path = r'E:\work\data\car20230705\mark\labels'

    source_dir = r'E:/work/data/car20230705/mark'
    file1_names = os.listdir(source_dir)
    for file1_name in file1_names:
        if '.' in file1_name or file1_name == 'images' or file1_name == 'labels' or file1_name == 'show':
            continue
        file2_names = os.listdir(source_dir + '/' + file1_name)
        for file2_name in file2_names:
            file3_names = os.listdir(source_dir + '/' + file1_name + '/' + file2_name)
            file3_label_names = []
            file3_image_names = []
            for file3_name in file3_names:
                # 去除21级标签不好的数据
                if 'lifan' in file1_name:
                    if '(21)' in file3_name and 'lifan20180402' not in file2_name:
                        continue
                elif 'meishan' in file1_name:
                    if '(21)' in file3_name:
                        continue
                elif 'shenlong' in file1_name:
                    if '(21)' in file3_name and 'shenlong20180315' not in file2_name:
                        continue
                elif 'xiaoqiche' in file1_name:
                    pass
                if file3_name.endswith('split'):
                    file3_image_names.append(file3_name)
                if file3_name.endswith('split_label'):
                    file3_label_names.append(file3_name)
            for file3_label_name in file3_label_names:
                print(file1_name, file2_name, file3_label_name)

                images_path_name = file3_label_name.replace('_label', '')
                images_source_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + images_path_name
                label_source_path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_label_name
                # 读取图片
                image_names = os.listdir(images_source_path)
                label_names = os.listdir(label_source_path)
                for image_name in tqdm(image_names):
                    label_name = image_name.split('.')[0] + '.txt'
                    if label_name not in label_names:
                        continue
                    image_source = images_source_path + '/' + image_name
                    label_source = label_source_path + '/' + label_name
                    image_target = image_path + '/index' + str(index) + '_' + image_name.replace('.tif', '.png')
                    label_target = label_path + '/index' + str(index) + '_' + label_name
                    index += 1
                    shutil.copy(image_source, image_target)
                    shutil.copy(label_source, label_target)


# 合并后图片中显示数据，查看标签像素点坐标是否正确
def merge_label2pic():
    image_path = r'E:\work\data\car20230705\mark\images'
    label_path = r'E:\work\data\car20230705\mark\labels'
    show_path = r'E:\work\data\car20230705\mark\show'
    if not os.path.exists(show_path):
        os.makedirs(show_path)

    # 读取图片
    image_names = os.listdir(image_path)
    label_names = os.listdir(label_path)
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
            label = arr[0]
            location.append([int(arr[1]), int(arr[2])])
            location.append([int(arr[3]), int(arr[4])])
            location.append([int(arr[5]), int(arr[6])])
            location.append([int(arr[7]), int(arr[8])])
            tmp_arr = np.array(location)
            if label == '0':
                contours.append(tmp_arr)
            elif label == '1':
                contours1.append(tmp_arr)

        img_contour = cv2.drawContours(img, contours, -1, (255, 255, 128), 1)
        img_contour = cv2.drawContours(img_contour, contours1, -1, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(show_path, image_name.replace('.tif', "_contour" + '.tif')), img_contour)


# 色差处理
def color_handle():
    image_path = r'E:\work\data\car20230705\mark\images'
    label_path = r'E:\work\data\car20230705\mark\labels'
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


# 色差处理
def color_handle_test():
    image_path = r'C:\Users\XQ\Desktop\temp\20230529'
    out_path = r'C:\Users\XQ\Desktop\temp\20230529_r1'
    image_file_names = os.listdir(image_path)
    img_suff = '.tif'
    for image_file_name in image_file_names:
        print(image_file_name)
        if not os.path.exists(out_path + '/' + image_file_name):
            os.makedirs(out_path + '/' + image_file_name)
        image_names = os.listdir(image_path + '/' + image_file_name)
        for image_name in tqdm(image_names):
            image_detail_path = image_path + '/' + image_file_name + '/' + image_name
            out_detail_path = out_path + '/' + image_file_name + '/' + image_name
            img = cv2.imread(image_detail_path)

            # ---------------------------------
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
            img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(image_detail_path.replace(img_suff, "_r1_" + img_suff), img2)
            cv2.imwrite(out_detail_path.replace(img_suff, "_r1_" + img_suff), img2)


# 生成train、val数据
# 训练模型
if __name__ == '__main__':
    color_handle_test()

