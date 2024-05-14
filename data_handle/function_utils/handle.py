# coding = utf8
from osgeo import gdal
import numpy as np
import cv2
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial.distance import pdist, squareform
import logging
import geopandas as gpd
import os
import math
logger = logging.getLogger()
logger.setLevel(logging.INFO)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp')
# 经纬度坐标转换为像素点坐标
def lon_lat_to_pixel(image_path, coordinates):
    dataset = gdal.Open(image_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    pixel_arr = []
    if not coordinates or len(coordinates) <= 1:
        pixel_arr.append([0, 0])
        pixel_arr.append([width, height])
    else:
        for coordinate in coordinates:
            coordinate_x = coordinate[0]
            coordinate_y = coordinate[1]
            location = [int((coordinate_x - geo_transform[0]) / geo_transform[1]),
                        int((coordinate_y - geo_transform[3]) / geo_transform[5])]
            pixel_arr.append(location)
    return pixel_arr


# 求像素点坐标最大值，最小值
def pixel_max_min(pixel_arr):
    pixel_x_min = -1
    pixel_x_max = -1
    pixel_y_min = -1
    pixel_y_max = -1
    for pixel in pixel_arr:
        pixel_x = pixel[0]
        pixel_y = pixel[1]
        if pixel_x_min == -1:
            pixel_x_min = pixel_x
        if pixel_y_min == -1:
            pixel_y_min = pixel_y
        if pixel_x_max == -1:
            pixel_x_max = pixel_x
        if pixel_y_max == -1:
            pixel_y_max = pixel_y

        if pixel_x > pixel_x_max:
            pixel_x_max = pixel_x
        if pixel_x < pixel_x_min:
            pixel_x_min = pixel_x
        if pixel_y > pixel_y_max:
            pixel_y_max = pixel_y
        if pixel_y < pixel_y_max:
            pixel_y_max = pixel_y
    return pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max


# 像素点坐标转换为经纬度坐标
def pixel_to_lon_lat(image_path, pixel_arr):
    dataset = gdal.Open(image_path)
    geo_transform = dataset.GetGeoTransform()
    coordinates = []
    for pixel in pixel_arr:
        pixel_x = pixel[0]
        pixel_y = pixel[1]
        location = [pixel_x * geo_transform[1] + geo_transform[0],
                    pixel_y * geo_transform[5] + geo_transform[3]]
        coordinates.append(location)
    return coordinates


# 切割图片
def split_image(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        picture = cv2.imread(image_path)
        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                pic = picture[h0: h1, w0: w1, :]
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width/2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    pic = picture[h0: h1, w0: w1, :]
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict

# 对图像进行数据增强
def apply_clahe_to_bgr_image(bgr_image, clip_limit=2, tile_grid_size=(8, 8)):
    # Convert the RGB image to Lab color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)

    # Split the Lab image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the L channel
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with the a and b channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert back to BGR color space
    bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2BGR)

    return bgr_image_clahe


def cut_image(picture, num_bands, w0, h0, w1, h1):
    data = [picture.GetRasterBand(band + 1).ReadAsArray(w0, h0, w1-w0, h1-h0) for band in range(num_bands)][::-1]
    pic = np.stack(data, axis=-1)
    return pic

def generate_cuts(pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max, cut_width, cut_height, offsets=(0,0)):
    for w in range(pixel_x_min + offsets[0], pixel_x_max, cut_width):
        for h in range(pixel_y_min + offsets[1], pixel_y_max, cut_height):
            w0, h0 = w, h
            w1, h1 = min(w0 + cut_width, pixel_x_max), min(h0 + cut_height, pixel_y_max)
            yield w0, h0, w1, h1

def split_image_large(image_paths, split_arr, augment=False, to_meter=False):
    split_images_dict = {}
    for image_path in image_paths:
        if image_path not in split_images_dict:
            split_images_dict[image_path] = {'split_images': [],
                                             'predict_results': {'all_box_arr': [], 'weight_arr': [], 'label_arr': [],
                                                                 'polygon_arr': [], 'mask_arr': []}}
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
        adfGeoTransform = picture.GetGeoTransform()
        if to_meter:
            ratio = pixel_2_meter(image_path)
            split_arr = [[int(j * ratio) for j in i] for i in split_arr]
        split_images_dict[image_path]['parameters'] = {'width': width, 'height': height, 'num_bands': num_bands, 'adfGeoTransform': adfGeoTransform}
        for cut_width, cut_height in split_arr:
            offsets = [(0, 0), (int(cut_width / 2), 0), (0, int(cut_height / 2)), (int(cut_width / 2), int(cut_height / 2))]
            for offset in offsets:
                for w0, h0, w1, h1 in generate_cuts(0, width, 0, height, cut_width, cut_height, offset):
                    if w1 > w0 and h1 > h0:  # Check if the cut dimensions are valid
                        pic = cut_image(picture, num_bands, w0, h0, w1, h1) # BGR Image
                        if augment:
                            pic = apply_clahe_to_bgr_image(pic)
                        split_images_dict[image_path]['split_images'].append({'image_path': image_path, 'location': [w0, h0, w1, h1], 'pic': pic})

    return split_images_dict
# 切割大图片
def split_image_large_legacy(image_path, split_arr, pixel_arr):
    pixel_x_min, pixel_x_max, pixel_y_min, pixel_y_max = pixel_max_min(pixel_arr)
    split_images_dict = {}
    num = 0
    for split_data in split_arr:
        # 要分割后的尺寸
        cut_width = split_data[0]
        cut_height = split_data[1]
        # 读取要分割的图片，以及其尺寸等数据
        # picture = cv2.imread(image_path)
        picture = gdal.Open(image_path)
        width, height, num_bands = picture.RasterXSize, picture.RasterYSize, picture.RasterCount
        # 计算可以划分的横纵的个数
        for w in range(pixel_x_min, pixel_x_max - 1, cut_width):
            for h in range(pixel_y_min, pixel_y_max - 1, cut_height):
                # 情况1
                w0 = w
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                # pic = picture[h0: h1, w0: w1, :]
                data = []
                for band in range(num_bands):
                    b = picture.GetRasterBand(band + 1)
                    data.append(b.ReadAsArray(w0, h0, w1-w0, h1-h0))
                # Assuming the image is RGB
                pic = np.zeros((h1-h0, w1-w0, num_bands), dtype=np.uint8)
                for b in range(num_bands):
                    pic[:, :, 0] = data[2]  # Blue channel
                    pic[:, :, 1] = data[1]  # Green channel
                    pic[:, :, 2] = data[0]  # Red channel
                split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                num += 1
                # 情况2
                w0 = w + int(cut_width / 2)
                h0 = h
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况3
                w0 = w
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1
                # 情况4
                w0 = w + int(cut_width/2)
                h0 = h + int(cut_height / 2)
                w1 = w0 + cut_width
                h1 = h0 + cut_height
                if w1 >= pixel_x_max:
                    w1 = pixel_x_max
                if h1 >= pixel_y_max:
                    h1 = pixel_y_max
                if w0 < pixel_x_max and h0 < pixel_y_max:
                    # pic = picture[h0: h1, w0: w1, :]
                    data = []
                    for band in range(num_bands):
                        b = picture.GetRasterBand(band + 1)
                        data.append(b.ReadAsArray(w0, h0, w1 - w0, h1 - h0))
                    # Assuming the image is RGB
                    pic = np.zeros((h1 - h0, w1 - w0, num_bands), dtype=np.uint8)
                    for b in range(num_bands):
                        pic[:, :, 0] = data[2]  # Blue channel
                        pic[:, :, 1] = data[1]  # Green channel
                        pic[:, :, 2] = data[0]  # Red channel
                    split_images_dict[num] = {'location': [w0, h0, w1, h1], 'pic': pic}
                    num += 1

    return split_images_dict


# 计算两多边形相交面积
def cal_area_2poly(data1, data2):
    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if point_num < 3:
        return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)

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

def geojson_to_shp(shp_dir):
    file_names = os.listdir(shp_dir)

    for file_name in file_names:
        if not file_name.endswith('.geojson'):
            continue
        target_dir = shp_dir
        target_filename = os.path.basename(file_name).split(".")[0]
        # Path to your GeoJSON file
        geojson_path = os.path.join(shp_dir, file_name)

        # Load the GeoJSON file into a GeoDataFrame
        gdf = gpd.read_file(geojson_path)

        # Path where you want to save the Shapefile
        shp_path = target_dir + '/' + target_filename + '.shp'

        # Save the GeoDataFrame as a Shapefile
        gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')

def change_img_path(image_paths):
    # 如果是文件就输出文件；如果是路径就输出路径下图片文件
    if os.path.isfile(image_paths):
        out_image_paths = [image_paths]
    else:
        out_image_paths = []
        image_names = os.listdir(image_paths)
        for image_name in image_names:
            if image_name.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(image_paths, image_name)
                out_image_paths.append(image_path)
    return out_image_paths

def pixel_2_meter(img_path):
    # Open the raster file using GDAL
    ds = gdal.Open(img_path)

    # Get raster size (width and height)
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Get georeferencing information
    geoTransform = ds.GetGeoTransform()
    pixel_size_x = geoTransform[1]  # Pixel width
    pixel_size_y = abs(geoTransform[5])  # Pixel height (absolute value)

    # Get the top latitude from the geotransform and the height
    # geoTransform[3] is the top left y, which gives the latitude
    latitude = geoTransform[3] - pixel_size_y * height
    # Close the dataset
    ds = None

    # Convert road width from meters to pixels
    # road_width_meters = line_width
    meters_per_degree = 111139 * math.cos(math.radians(latitude))
    thickness_pixels_ratio = 1 / (pixel_size_x * meters_per_degree)
    return thickness_pixels_ratio
def test():
    print('hello world')

if __name__ == '__main__':
    shp_dir = '/home/zkxq/develop/data/real_estate/out'
    geojson_to_shp(shp_dir)
