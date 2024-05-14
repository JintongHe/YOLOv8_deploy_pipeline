# coding=utf8
from osgeo import gdal
from ultralytics import YOLO
import cv2
import numpy as np
import time
import random
import yaml
import json
import logging
import os
import torch
import math
from shapely import Polygon, MultiPolygon
from tqdm import tqdm
import sys
import geopandas as gpd
sys.path.append('/home/zkxq/develop/develop/data_handle')
import function_utils.handle
import function_utils.special_handle

logger = logging.getLogger()
logger.setLevel(logging.INFO)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')

def get_config():
    config_path = 'config/wind_turbine_config.yaml'
    config_file = open(config_path, 'r', encoding='utf-8')
    file_info = config_file.read()
    config_dict = yaml.safe_load(file_info)
    return config_dict

def pre_handle(config_dict):
    image_paths = config_dict['image_paths']
    split_arr = config_dict['split_arr']
    # 图片切割
    logging.info('开始切割图片')
    split_images_dict = function_utils.handle.split_image_large(image_paths, split_arr, to_meter=True) # 米转为像素点
    logging.info("切割图片完成!!!")
    return split_images_dict

def model_predict(config_dict, split_images_dict):
    logging.info('开始模型预测')
    split_images_dict = function_utils.special_handle.model_predict_obb(config_dict, split_images_dict)
    logging.info('模型预测完成')
    return split_images_dict

def after_handle(config_dict, split_images_dict, method='area'):
    logging.info('开始后处理')
    overlap_percent = config_dict['overlap_percent']
    for image_path, split_image_dict in split_images_dict.items():

        predict_result = split_image_dict['predict_results']
        all_box_arr = predict_result['polygon_arr']
        weight_arr = predict_result['weight_arr']
        label_arr = predict_result['label_arr']

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'box': all_box_arr,
            'weight': weight_arr,
            'label': label_arr,
            'geometry': [Polygon(p) for p in all_box_arr]
        })

        # Spatial self-join to find overlapping polygons
        joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
        # print(joined_gdf.columns)
        # Initialize a set to keep track of processed indices
        processed_indices = set()

        for idx, row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
            row1 = idx
            row2 = row['index_right']

            if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
                continue

            poly1 = gdf.at[row1, 'geometry']
            poly2 = gdf.at[row2, 'geometry']
            area1 = poly1.area
            area2 = poly2.area
            over_area = poly1.intersection(poly2).area

            label1 = gdf.at[row1, 'label']
            label2 = gdf.at[row2, 'label']
            if label1 == label2 == 1:
                threshold = 0.5
                if over_area / area1 >= threshold or over_area / area2 >= threshold:
                    if area1 < area2:
                        processed_indices.add(row2)
                    else:
                        processed_indices.add(row1)
                continue
            if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
                # 取大的
                if method == 'area':
                    if area1 >= area2:
                        processed_indices.add(row2)
                    else:
                        processed_indices.add(row1)
                else:
                    if gdf['weight'].tolist()[row1] >= gdf['weight'].tolist()[row2]:
                        processed_indices.add(row2)
                    else:
                        processed_indices.add(row1)
        # Remove processed (merged) polygons
        gdf = gdf.drop(index=list(processed_indices))
        # Reconstruct the result
        split_images_dict[image_path]['res_results'] = {
            'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']],
            # Add other necessary fields
            'res_weight': gdf['weight'].tolist(),
            'res_label': gdf['label'].tolist()
        }
    logging.info('后处理完成')

    return split_images_dict

def create_geojson(config_dict, split_images_dict):
    logging.info('开始生成geojson')
    split_images_dict = function_utils.special_handle.create_geojson(config_dict, split_images_dict)
    logging.info('生成geojson完成')
    return split_images_dict

def main():
    logging.info('开始！！！')
    start_time = time.time()
    # 1，参数处理
    config_dict = get_config()
    is_test = False
    if is_test:
        image_path = "/home/zkxq/develop/data/wind_turbine"
        out_dir = "/home/zkxq/develop/data/wind_turbine/out"
    else:
        request_data = ''
        for i in range(len(sys.argv) - 1):
            request_data += sys.argv[i + 1]
        request_data = json.loads(request_data.replace('\'', '\"'))

        image_path = request_data['image_path']
        out_dir = request_data['out_dir']

    config_dict['start_time'] = start_time
    config_dict['image_paths'] = function_utils.handle.change_img_path(image_path)
    config_dict['out_file_path'] = out_dir

    # 2，预处理
    split_images_dict = pre_handle(config_dict)
    # 3，模型识别
    split_images_dict = model_predict(config_dict, split_images_dict)
    # 4，后处理
    split_images_dict = after_handle(config_dict, split_images_dict)
    # 5，生成输出结果
    split_images_dict = create_geojson(config_dict, split_images_dict)
    # 6，生成shape文件
    function_utils.handle.geojson_to_shp(out_dir)
    print(f'结果已存储在：{out_dir}')



if __name__ == '__main__':
    main()

