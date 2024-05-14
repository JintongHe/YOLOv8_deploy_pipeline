# coding = utf8
from osgeo import gdal, osr
import numpy as np
import cv2
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial.distance import pdist, squareform
from ultralytics import YOLO
from tqdm import tqdm
import random
import json
import os
import threading
import queue
import time
import torch
import geopandas as gpd
from ultralytics.models.sam import Predictor as SAMPredictor
gdal.AllRegister()

def towercrane_contour_generator(bbox, points, width):
    distances = squareform(pdist(points))
    farthest_points_idx = np.unravel_index(np.argmax(distances), distances.shape)
    p1, p2 = points[farthest_points_idx[0]], points[farthest_points_idx[1]]
    p1, p2 = intersection_points(bbox, p1, p2)
    # Distance between the two farthest points
    d = np.linalg.norm(p1 - p2)

    # Midpoint between the two farthest points
    midpoint = (p1 + p2) / 2

    # Direction vector
    dir_vector = (p2 - p1) / d

    # Construct the four vertices of the rectangle
    v1 = p1 + (width/2)*np.array([-dir_vector[1], dir_vector[0]])
    v2 = p1 + (width/2)*np.array([dir_vector[1], -dir_vector[0]])
    v3 = v2 + d*dir_vector
    v4 = v1 + d*dir_vector

    polygon = [list(v1), list(v2), list(v3), list(v4)]
    polygon = [[int(j) for j in i] for i in polygon]
    return polygon


def intersection_points(bbox, p1, p2):
    # Calculate the slope and y-intercept for the line passing through a and b
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    x3 = p1[0]
    y3 = p1[1]
    x4 = p2[0]
    y4 = p2[1]
    m = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    c = y3 - m * x3

    intersections = []

    # With x = x1 (left edge)
    y = m * x1 + c
    if y1 <= y <= y2:
        intersections.append((x1, y))

    # With x = x2 (right edge)
    y = m * x2 + c
    if y1 <= y <= y2:
        intersections.append((x2, y))

    # With y = y1 (top edge)
    if m != 0:
        x = (y1 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y1))

    # With y = y2 (bottom edge)
    if m != 0:
        x = (y2 - c) / m
        if x1 <= x <= x2:
            intersections.append((x, y2))
    intersections = list(set(intersections))
    a = np.array([int(intersections[0][0]), int(intersections[0][1])])
    b = np.array([int(intersections[1][0]), int(intersections[1][1])])
    return a, b

def model_predict_obb(config_dict, split_images_dict):
    model_path = config_dict['model_path']
    gpu_ids = config_dict['device']
    conf_thres = config_dict['conf_thres']
    polygon_threshold = config_dict['polygon_threshold']
    split_images_all = []
    for value in split_images_dict.values():
        split_images_all.extend(value['split_images'])

    subsets = [split_images_all[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    # Calculate the total number of images to be processed
    total_images = sum(len(subset) for subset in subsets)

    # Initialize the shared tqdm progress bar
    progress_bar = tqdm(total=total_images, desc="Processing Images")
    # A thread function to process a subset of images on a specific GPU
    def process_images(gpu_id, images_subset, output_queue, progress_bar):
        device = f'cuda:{gpu_id}'
        model = YOLO(model_path).to(device)
        batch_size = 16
        for i in range(0, len(images_subset), batch_size):
            sub_pics = [images_subset[idx]['pic'] for idx in range(i, min(i + batch_size, len(images_subset)))]
            results = model.predict(sub_pics, task='detect', save=False, conf=conf_thres,
                                    device=torch.device(f"cuda:{gpu_id}"),
                                    show_boxes=True,
                                    save_crop=False)
            for idx in range(i, min(i + batch_size, len(images_subset))):
                if results[idx - i] is None or results[idx - i].obb.data is None:
                    images_subset[idx]['result'] = None
                else:
                    images_subset[idx]['result'] = {'parameters': results[idx - i].obb.data.cpu(),
                                                    'boxes': results[idx - i].obb.xyxyxyxy.cpu()}
            # torch.cuda.empty_cache()
            progress_bar.update(len(sub_pics))
        output_queue.put(images_subset)  # Safely put the result into the queue

    # Create a queue to hold the results from each thread
    results_queue = queue.Queue()

    # Create and start a thread for each GPU/subset
    threads = []
    for i in range(len(gpu_ids)):
        if not i < len(subsets):
            continue
        gpu_id = gpu_ids[i]
        thread = threading.Thread(target=process_images, args=(gpu_id, subsets[i], results_queue, progress_bar))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Close the progress bar
    progress_bar.close()

    # Collect results from the queue
    all_results = []
    while not results_queue.empty():
        all_results.extend(results_queue.get())
    # {'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': pic, 'result': ___}

    for j in range(len(all_results)):
        x0 = all_results[j]['location'][0]
        y0 = all_results[j]['location'][1]
        x1 = all_results[j]['location'][2]
        y1 = all_results[j]['location'][3]
        result = all_results[j]['result']
        if not result:
            continue
        parameters = result['parameters']
        boxes = result['boxes']
        dim0, dim1, dim2 = boxes.shape
        for row in range(dim0):
            arr = boxes[row]
            parameter = parameters[row]
            location = []
            location.append([int(arr[0][0] + x0), int(arr[0][1] + y0)])
            location.append([int(arr[1][0] + x0), int(arr[1][1] + y0)])
            location.append([int(arr[2][0] + x0), int(arr[2][1] + y0)])
            location.append([int(arr[3][0] + x0), int(arr[3][1] + y0)])
            weight = float(parameter[-2])
            label = int(parameter[-1])

            image_path = all_results[j]['image_path']
            split_images_dict[image_path]['predict_results']['polygon_arr'].append(location)
            split_images_dict[image_path]['predict_results']['weight_arr'].append(weight)
            split_images_dict[image_path]['predict_results']['label_arr'].append(label)

    return split_images_dict

def model_predict(config_dict, split_images_dict):
    model_path = config_dict['model_path']
    gpu_ids = config_dict['device']
    conf_thres = config_dict['conf_thres']
    polygon_threshold = config_dict['polygon_threshold']
    split_images_all = []
    for value in split_images_dict.values():
        split_images_all.extend(value['split_images'])

    subsets = [split_images_all[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    # Calculate the total number of images to be processed
    total_images = sum(len(subset) for subset in subsets)

    # Initialize the shared tqdm progress bar
    progress_bar = tqdm(total=total_images, desc="Processing Images")
    # A thread function to process a subset of images on a specific GPU
    def process_images(gpu_id, images_subset, output_queue, progress_bar):
        device = f'cuda:{gpu_id}'
        model = YOLO(model_path).to(device)
        batch_size = 16
        for i in range(0, len(images_subset), batch_size):
            sub_pics = [images_subset[idx]['pic'] for idx in range(i, min(i + batch_size, len(images_subset)))]
            results = model.predict(sub_pics, task='segment', save=False, conf=conf_thres,
                                    device=torch.device(f"cuda:{gpu_id}"),
                                    show_boxes=True,
                                    save_crop=False)
            for idx in range(i, min(i + batch_size, len(images_subset))):
                if not list(results[idx - i].boxes.data):
                    images_subset[idx]['result'] = None
                else:
                    images_subset[idx]['result'] = {'masks': results[idx - i].masks.cpu(),
                                                    'boxes': results[idx - i].boxes.data.cpu()}
            # torch.cuda.empty_cache()
            progress_bar.update(len(sub_pics))
        output_queue.put(images_subset)  # Safely put the result into the queue

    # Create a queue to hold the results from each thread
    results_queue = queue.Queue()

    # Create and start a thread for each GPU/subset
    threads = []
    for i in range(len(gpu_ids)):
        if not i < len(subsets):
            continue
        gpu_id = gpu_ids[i]
        thread = threading.Thread(target=process_images, args=(gpu_id, subsets[i], results_queue, progress_bar))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Close the progress bar
    progress_bar.close()

    # Collect results from the queue
    all_results = []
    while not results_queue.empty():
        all_results.extend(results_queue.get())
    # {'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': pic, 'result': ___}

    for j in range(len(all_results)):
        x0 = all_results[j]['location'][0]
        y0 = all_results[j]['location'][1]
        x1 = all_results[j]['location'][2]
        y1 = all_results[j]['location'][3]
        result = all_results[j]['result']
        if not result:
            continue
        masks = result['masks'].xy
        boxes = result['boxes']
        dim0, dim1 = boxes.shape
        for row in range(dim0):
            arr = boxes[row]
            location = []
            location.append([int(arr[0] + x0), int(arr[1] + y0)])
            location.append([int(arr[2] + x0), int(arr[1] + y0)])
            location.append([int(arr[2] + x0), int(arr[3] + y0)])
            location.append([int(arr[0] + x0), int(arr[3] + y0)])
            # 去除边缘部分
            label = int(arr[-1])
            if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
                    location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
                continue
            weight = float(arr[-2])

            mask = np.array(masks[row]).astype(int)
            if mask.shape[0] <= 3:
                continue
            points = Polygon(mask).buffer(0)
            if type(points) is MultiPolygon:
                largest_polygon = max(points.geoms, key=lambda p: p.area)
                points = largest_polygon

            # area = pixel_area * points.area
            # print(area)
            # if area <= 200:
            #     continue
            exterior_coords_tuples = list(points.simplify(polygon_threshold, preserve_topology=True).exterior.coords)
            # Convert to a list of lists if required
            points = np.array([[int(t[0]), int(t[1])] for t in exterior_coords_tuples])
            if len(points.shape) < 2:
                continue
            points[:, 0] = points[:, 0] + x0
            points[:, 1] = points[:, 1] + y0

            mask[:, 0] = mask[:, 0] + x0
            mask[:, 1] = mask[:, 1] + y0
            image_path = all_results[j]['image_path']
            split_images_dict[image_path]['predict_results']['all_box_arr'].append(location)
            split_images_dict[image_path]['predict_results']['weight_arr'].append(weight)
            split_images_dict[image_path]['predict_results']['label_arr'].append(label)
            split_images_dict[image_path]['predict_results']['polygon_arr'].append(points.tolist())
            split_images_dict[image_path]['predict_results']['mask_arr'].append(mask.tolist())

    return split_images_dict

# def model_predict(config_dict, mid_dict):
#     model_path = config_dict['model_path']
#     device = config_dict['device']
#     conf_thres = config_dict['conf_thres']
#     polygon_threshold = config_dict['polygon_threshold']
#
#     split_images_dict = mid_dict['split_images_dict']
#
#     # 预测分割图片，并存储所有分割图像在大图中的标签
#     all_box_arr = []
#     weight_arr = []
#     label_arr = []
#     polygon_arr = []
#     mask_arr = []
#     pics = [sub_dict['pic'] for sub_dict in split_images_dict.values()]
#     model = YOLO(model_path)
#     batch_size = 16
#     for i in tqdm(range(0, len(pics), batch_size)):
#         results = None
#         sub_pics = pics[i:i+batch_size]
#         results = model.predict(sub_pics, task='segment', save=False, conf=conf_thres,
#                                 device=device,
#                                 boxes=True,
#                                 save_crop=False)
#         for j in range(i, min(i+batch_size, len(pics))):
#             x0 = split_images_dict[j]['location'][0]
#             y0 = split_images_dict[j]['location'][1]
#             x1 = split_images_dict[j]['location'][2]
#             y1 = split_images_dict[j]['location'][3]
#             if len(list(results[j-i].boxes.data)) == 0:
#                 continue
#             masks = results[j-i].masks.xy
#             boxes = results[j-i].boxes.data
#             dim0, dim1 = boxes.shape
#             for row in range(dim0):
#                 arr = boxes[row]
#                 location = []
#                 location.append([int(arr[0] + x0), int(arr[1] + y0)])
#                 location.append([int(arr[2] + x0), int(arr[1] + y0)])
#                 location.append([int(arr[2] + x0), int(arr[3] + y0)])
#                 location.append([int(arr[0] + x0), int(arr[3] + y0)])
#                 # 去除边缘部分
#                 if (location[0][0] <= x0 + 2 or location[1][0] >= x1 - 2 or
#                         location[0][1] <= y0 + 2 or location[2][1] >= y1 - 2):
#                     continue
#                 weight = float(arr[-2])
#                 label = int(arr[-1])
#
#                 mask = np.array(masks[row]).astype(int)
#                 if mask.shape[0] <= 3:
#                     continue
#                 points = Polygon(mask).buffer(0)
#                 if type(points) is MultiPolygon:
#                     largest_polygon = max(points.geoms, key=lambda p: p.area)
#                     points = largest_polygon
#                 exterior_coords_tuples = list(points.simplify(polygon_threshold, preserve_topology=True).exterior.coords)
#                 # Convert to a list of lists if required
#                 points = np.array([[int(t[0]), int(t[1])] for t in exterior_coords_tuples])
#                 points[:, 0] = points[:, 0] + x0
#                 points[:, 1] = points[:, 1] + y0
#
#                 mask[:, 0] = mask[:, 0] + x0
#                 mask[:, 1] = mask[:, 1] + y0
#
#                 all_box_arr.append(location)
#                 weight_arr.append(weight)
#                 label_arr.append(label)
#                 polygon_arr.append(points.tolist())
#                 mask_arr.append(mask.tolist())
#         del results
#         torch.cuda.empty_cache()
#
#     predict_result = {'all_box_arr': all_box_arr, 'weight_arr': weight_arr, 'label_arr': label_arr,
#                       'polygon_arr': polygon_arr, 'mask_arr': mask_arr}
#     mid_dict['predict_result'] = predict_result
#     # result = [all_box_arr, weight_arr, label_arr, polygon_arr, mask_arr]
#     return mid_dict

# 去重（融合Polygon的方法）
def after_handle_merge_polygon(config_dict, split_images_dict):
    overlap_percent = config_dict['overlap_percent']
    for image_path, split_image_dict in split_images_dict.items():
        predict_result = split_image_dict['predict_results']
        all_box_arr = predict_result['all_box_arr']
        weight_arr = predict_result['weight_arr']
        label_arr = predict_result['label_arr']
        polygon_arr = predict_result['polygon_arr']
        mask_arr = predict_result['mask_arr']

        # Pre-calculate polygons and their areas
        polygons = [Polygon(p).buffer(0) for p in polygon_arr]
        areas = np.array([p.area for p in polygons])

        box_len = len(polygons)
        flags = np.zeros(box_len, dtype=int)

        for row1 in tqdm(range(box_len - 1)):
            if flags[row1]:
                continue
            for row2 in range(row1 + 1, box_len):
                if flags[row2]:
                    continue
                if polygons[row1].intersects(polygons[row2]):
                    inter_area = polygons[row1].intersection(polygons[row2]).area
                    if inter_area / areas[row1] >= overlap_percent or inter_area / areas[row2] >= overlap_percent:
                        flags[row1] = 1
                        union_poly = polygons[row1].union(polygons[row2]).buffer(0)
                        if isinstance(union_poly, MultiPolygon):
                            union_poly = max(union_poly.geoms, key=lambda p: p.area)
                        polygons[row2] = union_poly
                        areas[row2] = union_poly.area
                        union_poly = list(union_poly.exterior.coords)
                        union_poly = np.array([[int(t[0]), int(t[1])] for t in union_poly])
                        # Update the polygon array directly if needed
                        x_min = np.min(union_poly, axis=0)[0]
                        x_max = np.max(union_poly, axis=0)[0]
                        y_min = np.min(union_poly, axis=0)[1]
                        y_max = np.max(union_poly, axis=0)[1]
                        all_box_arr[row2] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                        polygon_arr[row2] = union_poly.tolist()
                        mask_arr[row2] = mask_arr[row2] + mask_arr[row1]
                        mask_arr[row1] = []

        # Build the result based on flags
        valid_indices = np.where(flags == 0)[0]
        res_result = {key[1]: [predict_result[key[0]][i] for i in valid_indices] for key in
                                  [['all_box_arr', 'res_box'], ['weight_arr', 'res_weight'], ['label_arr', 'res_label'],
                                   ['polygon_arr', 'res_polygon'], ['mask_arr', 'res_mask']]}

        split_images_dict[image_path]['res_results'] = res_result
    return split_images_dict
# def after_handle_merge_polygon(config_dict, mid_dict):
#     predict_result = mid_dict['predict_result']
#     all_box_arr = predict_result['all_box_arr']
#     weight_arr = predict_result['weight_arr']
#     label_arr = predict_result['label_arr']
#     polygon_arr = predict_result['polygon_arr']
#     mask_arr = predict_result['mask_arr']
#
#     overlap_percent = config_dict['overlap_percent']
#     # 干掉重合的预测结果
#     box_len = len(polygon_arr)
#     flags = [0 for x in range(box_len)]
#     for row1 in range(box_len - 1):
#         if flags[row1] == 1:
#             continue
#         poly1 = Polygon(polygon_arr[row1]).buffer(0)
#         for row2 in range(row1 + 1, box_len):
#             if flags[row2] == 1:
#                 continue
#             poly2 = Polygon(polygon_arr[row2]).buffer(0)
#
#             area1 = poly1.area
#             area2 = poly2.area
#             if not poly1.intersects(poly2):
#                 continue
#             over_area = poly1.intersection(poly2).area
#             if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
#                 flags[row1] = 1
#                 union_poly = poly1.union(poly2)
#                 if not union_poly.is_valid:
#                     union_poly = union_poly.buffer(0)
#                 if type(union_poly) is MultiPolygon:
#                     # points = points.geoms[0]
#                     union_poly = max(union_poly.geoms, key=lambda p: p.area)
#                     # if type(largest_polygon) is MultiPolygon:
#                     #     largest_polygon = largest_polygon.geoms[0]
#                 union_poly = list(union_poly.exterior.coords)
#                 # Convert to a list of lists if required
#                 union_poly = np.array([[int(t[0]), int(t[1])] for t in union_poly])
#                 x_min = np.min(union_poly, axis=0)[0]
#                 x_max = np.max(union_poly, axis=0)[0]
#                 y_min = np.min(union_poly, axis=0)[1]
#                 y_max = np.max(union_poly, axis=0)[1]
#                 all_box_arr[row2] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
#                 polygon_arr[row2] = union_poly.tolist()
#                 mask_arr[row2] = mask_arr[row2] + mask_arr[row1]
#                 mask_arr[row1] = []
#     mid_dict['flags'] = flags
#     res_box = []
#     res_weight = []
#     res_label = []
#     res_polygon = []
#     res_mask = []
#     for index in range(len(flags)):
#         if flags[index] == 1:
#             continue
#         elif flags[index] == 0:
#             res_box.append(all_box_arr[index])
#             res_weight.append(weight_arr[index])
#             res_label.append(label_arr[index])
#             res_polygon.append(polygon_arr[index])
#             res_mask.append(mask_arr[index])
#     res_result = {'res_box': res_box, 'res_weight': res_weight, 'res_label': res_label, 'res_polygon': res_polygon,
#                   'res_mask': res_mask}
#     mid_dict['res_result'] = res_result
#     return mid_dict

# 去重（锚框去重方法）
def after_handle_bbox(config_dict, split_images_dict, method='area'):
    overlap_percent = config_dict['overlap_percent']
    for image_path, split_image_dict in split_images_dict.items():
        predict_results = split_image_dict['predict_results']
        all_box_arr = predict_results['all_box_arr']
        weight_arr = predict_results['weight_arr']
        label_arr = predict_results['label_arr']
        polygon_arr = predict_results['polygon_arr']
        mask_arr = predict_results['mask_arr']

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'polygon': polygon_arr,
            'weight': weight_arr,
            'label': label_arr,
            'mask': mask_arr,
            'box': all_box_arr,
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
            'res_box': gdf['box'].tolist(),
            # Add other necessary fields
            'res_weight': gdf['weight'].tolist(),
            'res_label': gdf['label'].tolist(),
            'res_mask': gdf['mask'].tolist(),
            'res_polygon': gdf['polygon'].tolist()
        }
    return split_images_dict
# def after_handle_bbox(config_dict, mid_dict, method='area'):
#     predict_result = mid_dict['predict_result']
#     all_box_arr = predict_result['all_box_arr']
#     weight_arr = predict_result['weight_arr']
#     label_arr = predict_result['label_arr']
#     polygon_arr = predict_result['polygon_arr']
#     mask_arr = predict_result['mask_arr']
#
#     overlap_percent = config_dict['overlap_percent']
#
#     # Create GeoDataFrame
#     gdf = gpd.GeoDataFrame({
#         'polygon': polygon_arr,
#         'weight': weight_arr,
#         'label': label_arr,
#         'mask': mask_arr,
#         'box': all_box_arr,
#         'geometry': [Polygon(p) for p in all_box_arr]
#     })
#
#     # Spatial self-join to find overlapping polygons
#     joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
#     # print(joined_gdf.columns)
#     # Initialize a set to keep track of processed indices
#     processed_indices = set()
#
#     for idx, row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
#         row1 = idx
#         row2 = row['index_right']
#
#         if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
#             continue
#
#         poly1 = gdf.at[row1, 'geometry']
#         poly2 = gdf.at[row2, 'geometry']
#         area1 = poly1.area
#         area2 = poly2.area
#         over_area = poly1.intersection(poly2).area
#
#         if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
#             # 取大的
#             if method == 'area':
#                 if area1 >= area2:
#                     processed_indices.add(row2)
#                 else:
#                     processed_indices.add(row1)
#             else:
#                 if gdf['weight'].tolist()[row1] >= gdf['weight'].tolist()[row2]:
#                     processed_indices.add(row2)
#                 else:
#                     processed_indices.add(row1)
#     # Remove processed (merged) polygons
#     gdf = gdf.drop(index=list(processed_indices))
#     # Reconstruct the result
#     mid_dict['res_result'] = {
#         'res_box': gdf['box'].tolist(),
#         # Add other necessary fields
#         'res_weight': gdf['weight'].tolist(),
#         'res_label': gdf['label'].tolist(),
#         'res_mask': gdf['mask'].tolist(),
#         'res_polygon': gdf['polygon'].tolist()
#     }
#
#     return mid_dict

# 去重（多边形去重方法）
def after_handle_polygon(config_dict, split_images_dict, method='area'):
    overlap_percent = config_dict['overlap_percent']
    for image_path, split_image_dict in split_images_dict.items():
        predict_results = split_image_dict['predict_results']
        all_box_arr = predict_results['all_box_arr']
        weight_arr = predict_results['weight_arr']
        label_arr = predict_results['label_arr']
        polygon_arr = predict_results['polygon_arr']
        mask_arr = predict_results['mask_arr']
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'box': all_box_arr,
            'weight': weight_arr,
            'label': label_arr,
            'mask': mask_arr,
            'geometry': [Polygon(p) for p in polygon_arr]
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
            'res_box': gdf['box'].tolist(),
            # Add other necessary fields
            'res_weight': gdf['weight'].tolist(),
            'res_label': gdf['label'].tolist(),
            'res_mask': gdf['mask'].tolist(),
            'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']]
        }
    return split_images_dict
# def after_handle_polygon(config_dict, mid_dict, method='area'):
#     predict_result = mid_dict['predict_result']
#     all_box_arr = predict_result['all_box_arr']
#     weight_arr = predict_result['weight_arr']
#     label_arr = predict_result['label_arr']
#     polygon_arr = predict_result['polygon_arr']
#     mask_arr = predict_result['mask_arr']
#
#     overlap_percent = config_dict['overlap_percent']
#     # Create GeoDataFrame
#     gdf = gpd.GeoDataFrame({
#         'box': all_box_arr,
#         'weight': weight_arr,
#         'label': label_arr,
#         'mask': mask_arr,
#         'geometry': [Polygon(p) for p in polygon_arr]
#     })
#
#     # Spatial self-join to find overlapping polygons
#     joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
#     # print(joined_gdf.columns)
#     # Initialize a set to keep track of processed indices
#     processed_indices = set()
#
#     for idx, row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
#         row1 = idx
#         row2 = row['index_right']
#
#         if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
#             continue
#
#         poly1 = gdf.at[row1, 'geometry']
#         poly2 = gdf.at[row2, 'geometry']
#         area1 = poly1.area
#         area2 = poly2.area
#         over_area = poly1.intersection(poly2).area
#
#         if over_area / area1 >= overlap_percent or over_area / area2 >= overlap_percent:
#             # 取大的
#             if method == 'area':
#                 if area1 >= area2:
#                     processed_indices.add(row2)
#                 else:
#                     processed_indices.add(row1)
#             else:
#                 if gdf['weight'].tolist()[row1] >= gdf['weight'].tolist()[row2]:
#                     processed_indices.add(row2)
#                 else:
#                     processed_indices.add(row1)
#     # Remove processed (merged) polygons
#     gdf = gdf.drop(index=list(processed_indices))
#     # Reconstruct the result
#     mid_dict['res_result'] = {
#         'res_box': gdf['box'].tolist(),
#         # Add other necessary fields
#         'res_weight': gdf['weight'].tolist(),
#         'res_label': gdf['label'].tolist(),
#         'res_mask': gdf['mask'].tolist(),
#         'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']]
#     }
#
#     return mid_dict

def after_handle_obb(config_dict, split_images_dict, method='area'):
    overlap_percent = config_dict['overlap_percent']
    for image_path, split_image_dict in split_images_dict.items():
        predict_results = split_image_dict['predict_results']
        weight_arr = predict_results['weight_arr']
        label_arr = predict_results['label_arr']
        polygon_arr = predict_results['polygon_arr']
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'weight': weight_arr,
            'label': label_arr,
            'geometry': [Polygon(p) for p in polygon_arr]
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
            'res_weight': gdf['weight'].tolist(),
            'res_label': gdf['label'].tolist(),
            'res_polygon': [[[int(t[0]), int(t[1])] for t in list(poly.exterior.coords)] for poly in gdf['geometry']]
        }
    return split_images_dict

def sam_handle(config_dict, mid_dict, method='bbox'):
    orig_img = config_dict['image_path']
    # SAM Parameters
    conf = 0.25
    crop_n_layers = 0
    crop_overlap_ratio = 512 / 1500
    crop_downscale_factor = 1
    point_grids = None
    points_stride = 32
    points_batch_size = 64
    conf_thres = 0.88
    stability_score_thresh = 0.95
    stability_score_offset = 0.95
    crops_nms_thresh = 0.7

    # Create SAMPredictor
    overrides = dict(conf=conf, task='segment', save=False, mode='predict', imgsz=1024,
                     model="/home/zkxq/develop/models/sam_h.pt")
    predictor = SAMPredictor(overrides=overrides)

    predict_result = mid_dict['res_result']
    all_box_arr = predict_result['res_box']
    # weight_arr = predict_result['weight_arr']
    # label_arr = predict_result['label_arr']
    polygon_arr = predict_result['res_polygon']

    sam_boxes_list = []
    for i in range(len(all_box_arr)):
        one_poly = Polygon(polygon_arr[i])
        if not one_poly.is_valid:
            one_poly = one_poly.buffer(0)
        centroid = one_poly.centroid
        one_box = all_box_arr[i]
        sam_box = [one_box[0][0], one_box[0][1], one_box[2][0], one_box[2][1]]
        sam_boxes_list.append((sam_box, [[int(centroid.x), int(centroid.y)]]))
    # Set image
    sam_masks = []
    predictor.set_image(cv2.imread(orig_img))  # set with np.ndarray
    for sam_box, sam_point in sam_boxes_list:
        if method == 'bbox':
            results = predictor(points=None, labels=None, bboxes=sam_box, crop_n_layers=crop_n_layers,
                                crop_overlap_ratio=crop_overlap_ratio,
                                crop_downscale_factor=crop_downscale_factor, point_grids=point_grids,
                                points_stride=points_stride,
                                points_batch_size=points_batch_size, conf_thres=conf_thres,
                                stability_score_thresh=stability_score_thresh,
                                stability_score_offset=stability_score_offset,
                                crops_nms_thresh=crops_nms_thresh)
        else:
            results = predictor(points=sam_point, labels=[1], bboxes=None, crop_n_layers=crop_n_layers,
                                crop_overlap_ratio=crop_overlap_ratio,
                                crop_downscale_factor=crop_downscale_factor, point_grids=point_grids,
                                points_stride=points_stride,
                                points_batch_size=points_batch_size, conf_thres=conf_thres,
                                stability_score_thresh=stability_score_thresh,
                                stability_score_offset=stability_score_offset,
                                crops_nms_thresh=crops_nms_thresh)
        sam_mask = results[0].masks.xy[0]
        sam_mask.astype(int).tolist()
        sam_mask = Polygon(sam_mask).buffer(0)
        if type(sam_mask) is MultiPolygon:
            sam_mask = max(sam_mask.geoms, key=lambda p: p.area)
        tolerance = 0.5
        sam_mask = sam_mask.simplify(tolerance, preserve_topology=True)
        # Get the exterior coordinates of the polygon as a list of tuples
        exterior_coords_tuples = list(sam_mask.exterior.coords)
        # Convert to a list of lists if required
        sam_mask = [[int(t[0]), int(t[1])] for t in exterior_coords_tuples]
        sam_masks.append(sam_mask)

    # Reset image
    predictor.reset_image()
    mid_dict['res_results']['res_sam_mask'] = sam_masks
    return mid_dict

def create_geojson(config_dict, split_images_dict):
    # Variables moved outside the loop since their values are constant per function call
    start_time = config_dict['start_time']
    # out_flag = config_dict['out_flag']
    out_file_path = config_dict['out_file_path']
    class_names = config_dict['class_dict']
    os.makedirs(out_file_path, exist_ok=True)
    for image_path, split_image_dict in split_images_dict.items():
        res_results = split_image_dict['res_results']
        adfGeoTransform = split_image_dict['parameters']['adfGeoTransform']

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        time_now = int(time.time())
        out_name = f"{image_name}_{str(time_now)[:5]}.geojson"

        date = image_name.split('.')[0][-8:]

        res_dict = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": []
        }

        for polygon, label, weight in zip(res_results['res_polygon'], res_results['res_label'], res_results['res_weight']):
            name = class_names[label]
            coordinates = [[xy[0] * adfGeoTransform[1] + adfGeoTransform[0], xy[1] * adfGeoTransform[5] + adfGeoTransform[3]] for xy in polygon]
            coordinates.append(coordinates[0])  # Close the polygon

            feature = {
                "type": "Feature",
                "properties": {"Id": 0, "name": name, "date": date, "area": 0.0, "label": label, "result": 1,
                               "XMMC": "", "HYMC": "", "weight": weight, "bz": 0},
                "geometry": {"type": "Polygon", "coordinates": [coordinates]}
            }
            res_dict['features'].append(feature)

        consume_time = time.time() - start_time
        res_dict['consume_time'] = consume_time
        split_image_dict['res'] = json.dumps(res_dict)  # Convert the dictionary to a JSON string
        # mid_dict['res'] = json.dumps(res_dict)  # Convert the dictionary to a JSON string

        # if out_flag:
        # Use json.dump to write the JSON data directly to a file
        with open(os.path.join(out_file_path, out_name), 'w', encoding='utf8') as out_file:
            json.dump(res_dict, out_file, ensure_ascii=False)

    return split_images_dict

# def create_geojson(config_dict, mid_dict):
#     start_time = config_dict['start_time']
#     image_path = config_dict['image_path']
#     out_flag = config_dict['out_flag']
#     out_file_path = config_dict['out_file_path']
#     class_names = config_dict['class_dict']
#
#     res_result = mid_dict['res_result']
#     res_box = res_result['res_box']
#     res_weight = res_result['res_weight']
#     res_label = res_result['res_label']
#     res_polygon = res_result['res_polygon']
#
#     image_name = image_path.split('/')[-1]
#     time_now = int(time.time())
#     out_name = image_name.split('.')[0] + '_' + str(time_now)[0:5] + '.geojson'
#
#     gdal.AllRegister()
#     dataset = gdal.Open(image_path)
#     adfGeoTransform = dataset.GetGeoTransform()
#
#     res_dict = {
#         "type": "FeatureCollection",
#         "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
#         "features": []
#     }
#     date = image_name.split('.')[0][-8:]
#     for index in range(len(res_box)):
#         polygon = res_polygon[index]
#         label = res_label[index]
#         weight = res_weight[index]
#         name = class_names[label]
#         feature = {"type": "Feature",
#                    "properties": {"Id": 0, "name": name, "date": date, "area": 0.0, "label": label, "result": 1,
#                                   "XMMC": "", "HYMC": "", "weight": weight, "bz": 0},
#                    "geometry": {"type": "Polygon", "coordinates": []}}
#         coordinate = []
#         for xy in polygon:
#             location = [xy[0] * adfGeoTransform[1] + adfGeoTransform[0],
#                         xy[1] * adfGeoTransform[5] + adfGeoTransform[3]]
#             coordinate.append(location)
#         coordinate.append(coordinate[0])
#         feature['geometry']['coordinates'].append(coordinate)
#         res_dict['features'].append(feature)
#
#     end_time = time.time()
#     consume_time = end_time - start_time
#     res = str(res_dict).replace('\'', '"').replace('None', '"None"')
#     res_dict['consume_time'] = consume_time
#     mid_dict['res'] = res
#
#     # 输出json文件， 默认不输出
#     if out_flag:
#         out_file = open(out_file_path + '/' + out_name, 'w', encoding='utf8')
#         out_file.write(res)
#         out_file.close()
#
#     return mid_dict

def reproject_dataset(src_ds, dst_srs):
    """Reproject a dataset to a new spatial reference system."""
    # Define target SRS
    dst_wkt = dst_srs.ExportToWkt()

    # Set up the transformation
    resampling_method = gdal.GRA_Bilinear
    error_threshold = 0.125  # Error threshold for transformation approximation
    warp_options = gdal.WarpOptions(resampleAlg=resampling_method, dstSRS=dst_wkt, errorThreshold=error_threshold)

    # Perform the reprojection
    reprojected_ds = gdal.Warp('temp.tif', src_ds, options=warp_options, format='MEM')

    return reprojected_ds


def calculate_area(geotiff_path):
    # Open the GeoTIFF file
    src_ds = gdal.Open(geotiff_path)

    # Define the target SRS (e.g., UTM)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(32633)  # Example: UTM zone 33N. Change as needed.

    # Reproject the dataset
    reprojected_ds = reproject_dataset(src_ds, target_srs)

    # Get the GeoTransform and calculate pixel size
    geotransform = reprojected_ds.GetGeoTransform()
    pixelWidth = geotransform[1]
    pixelHeight = -geotransform[5]  # Pixel height is generally negative

    # Calculate area of each pixel
    pixelArea = pixelWidth * pixelHeight

    # Get raster size
    rasterXSize = reprojected_ds.RasterXSize
    rasterYSize = reprojected_ds.RasterYSize

    # Calculate total area
    # totalArea = rasterXSize * rasterYSize * pixelArea

    return pixelArea