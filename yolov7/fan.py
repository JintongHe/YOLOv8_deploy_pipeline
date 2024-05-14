# python test.py --img 640 --batch 16 --data data/fan.yaml --weights fan_models/Fan_Jintong_Epoch_124.pt
# python detect.py --weights runs/train/Fan/best2.pt --source inference/Satellite_Images/out/20221119_split --device 0 --save-txt
# python train.py --img 640 --batch 32 --epochs 300 --data data/fan.yaml --weights yolov7.pt --name Wind_Turbine_Detection
# python train.py --workers 8 --device 0 --batch-size 32 --data data/fan.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights [string]::Empty --name yolov7 --hyp data/hyp.scratch.p5.yaml
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import shutil
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from osgeo import gdal
import os
import cv2
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
import time


def compute_polygon_area(points):
    point_num = len(points)
    if (point_num < 3): return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


def cal_area_2poly(data1, data2):
    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


def de_results(all_box_arr, weight_arr):
    print('开始后处理')
    # 干掉重合的预测结果
    box_len = len(all_box_arr)
    flags = [0 for x in range(box_len)]
    for row1 in range(box_len - 1):
        if flags[row1] == 1:
            continue
        arr1 = all_box_arr[row1]
        weight1 = weight_arr[row1]
        for row2 in range(row1 + 1, box_len):

            if flags[row2] == 1:
                continue
            arr2 = all_box_arr[row2]
            if arr1[0][0] > arr2[1][0] or arr1[1][0] < arr2[0][0]:
                continue
            if arr1[0][1] > arr2[2][1] or arr1[2][1] < arr2[0][1]:
                continue

            area1 = compute_polygon_area(arr1)
            area2 = compute_polygon_area(arr2)
            weight2 = weight_arr[row2]
            over_area = cal_area_2poly(arr1, arr2)
            if over_area / area1 >= 0.65 or over_area / area2 >= 0.65:
                if weight1 >= weight2:
                    flags[row2] = 1
                else:
                    flags[row1] = 1
    print('后处理完成')
    return flags


def split_image(image_path, split_arr):
    print('开始切割图片')

    image_name = image_path.split('/')[-1]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    split_out_path = path + '/out/' + image_name.replace('.' + suf, '_split')
    if not os.path.exists(split_out_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(split_out_path)
    else:
        shutil.rmtree(split_out_path)
        os.makedirs(split_out_path)

    num = 0
    for split_data in split_arr:
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
                # pic = cv2.resize(pic, (608, 608))
                result_path = split_out_path + '/split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                cv2.imwrite(result_path, pic)
                # 多重切割处理
                if i * cut_width + cut_width < width:
                    pic1 = picture[j * cut_height: (j + 1) * cut_height,
                           i * cut_width + int(cut_width / 2): (i + 1) * cut_width + int(cut_width / 2), :]
                    # pic1 = cv2.resize(pic1, (608, 608))
                    result_path1 = split_out_path + '/r1split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
                if j * cut_height + cut_height < height:
                    pic1 = picture[j * cut_height + int(cut_height / 2): (j + 1) * cut_height + int(cut_height / 2),
                           i * cut_width: (i + 1) * cut_width, :]
                    # pic1 = cv2.resize(pic1, (608, 608))
                    result_path1 = split_out_path + '/r2split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
                if i * cut_width + cut_width < width and j * cut_height + cut_height < height:
                    pic1 = picture[j * cut_height + int(cut_height / 2): (j + 1) * cut_height + int(cut_height / 2),
                           i * cut_width + int(cut_width / 2): (i + 1) * cut_width + int(cut_width / 2), :]
                    # pic1 = cv2.resize(pic1, (608, 608))
                    result_path1 = split_out_path + '/r3split' + str(num) + ('_{}_{}.' + suf).format(i + 1, j + 1)
                    cv2.imwrite(result_path1, pic1)
        num += 1

    print("切割图片完成!!!")
    return split_out_path


def inference(weights, source, device='0', imgsz=640, conf_thres=0.6, iou_thres=0.45):
    all_box_arr = []
    weight_arr = []
    label_arr = []
    image_paths = []

    # Directories
    save_dir = Path(increment_path(Path('runs/detect') / 'exp', exist_ok=False))  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Trace model
    model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label_arr.append(int(cls))
                    xyxy = [int(i) for i in xyxy]
                    all_box_arr.append([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]],
                                        [xyxy[0], xyxy[3]]])  # label format in clockwise
                    weight_arr.append(round(conf.item(), 2))
                    # print(source + '/' + str(p).split('\\')[-1])
                    image_paths.append(source + '/' + str(p).split('\\')[-1])

                    # print('/'.join(str(p).split('\\')[-5:]))

            # Print time (inference + NM S)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')
    return all_box_arr, weight_arr, label_arr, image_paths


def predict_thread_run(raw_boxes, raw_image_paths, raw_labels, raw_weights, split_arr):
    # 预测分割图片，并存储所有分割图像在大图中的标签
    all_box_arr = []
    weight_arr = []
    label_arr = []
    print(len(raw_image_paths), 'len(image_data)')
    for i in range(len(raw_image_paths)):
        image_path = raw_image_paths[i]
        split_image_name = image_path.split("/")[-1]
        # Predict on an image

        name_arr = split_image_name.split('.')[0].split('_')
        # 当前图第一个像素点在原大图中的坐标
        # base_name = name_arr[0] + '_1_1.' + suf
        split_type = int(name_arr[0].split('split')[1])
        w0 = split_arr[split_type][0]
        h0 = split_arr[split_type][1]
        x0 = w0 * (int(name_arr[1]) - 1)
        y0 = h0 * (int(name_arr[2]) - 1)

        img1 = cv2.imread(image_path)
        w1 = img1.shape[1]
        h1 = img1.shape[0]
        x1 = w0 * (int(name_arr[1]) - 1) + w1
        y1 = h0 * (int(name_arr[2]) - 1) + h1

        if 'r1split' in name_arr[0]:
            x0 = w0 * (int(name_arr[1]) - 1) + w0 / 2
            y0 = h0 * (int(name_arr[2]) - 1)
        if 'r2split' in name_arr[0]:
            x0 = w0 * (int(name_arr[1]) - 1)
            y0 = h0 * (int(name_arr[2]) - 1) + h0 / 2
        if 'r3split' in name_arr[0]:
            x0 = w0 * (int(name_arr[1]) - 1) + w0 / 2
            y0 = h0 * (int(name_arr[2]) - 1) + h0 / 2

        # boxes = raw_boxes
        # dim0 = len(boxes)
        # for row in range(dim0):
        arr = raw_boxes[i]
        location = [[int(arr[0][0] + x0), int(arr[0][1] + y0)], [int(arr[1][0] + x0), int(arr[1][1] + y0)],
                    [int(arr[2][0] + x0), int(arr[2][1] + y0)], [int(arr[3][0] + x0), int(arr[3][1] + y0)]]
        # 去除边缘部分
        if location[0][0] == x0:
            continue
        if location[1][0] == x1:
            continue
        if location[0][1] == y0:
            continue
        if location[2][1] == y1:
            continue
        weight = raw_weights[i]
        label = raw_labels[i]

        all_box_arr.append(location)
        weight_arr.append(weight)
        label_arr.append(label)
    print('坐标变换完成')

    # result = [all_box_arr, weight_arr, label_arr]
    return all_box_arr, weight_arr, label_arr


def show_one_image(image_path, res_box, res_weight, res_label):
    image_name = image_path.split('/')[-1]
    path = image_path.replace('/' + image_name, '')
    suf = image_name.split('.')[1]
    show_path = path + '/out/' + image_name.replace('.' + suf, '_show')
    # 大图打上标签
    if not os.path.exists(show_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(show_path)
    image_name = image_path.split('/')[-1]
    suf = image_name.split('.')[1]
    img = cv2.imread(image_path)
    box_len = len(res_box)
    colors = [random.randint(0, 255) for _ in range(3)]
    for i in range(box_len):
        label = res_label[i]
        confidence = res_weight[i]
        xyxy = [res_box[i][0][0], res_box[i][0][1], res_box[i][2][0], res_box[i][2][1]]
        plot_one_box(xyxy, img, label=f'{label} {confidence}', color=colors, line_thickness=2)
    cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour." + suf)), img)
    print('已生成图片保存至{}'.format(os.path.join(show_path, image_name.replace('.' + suf, "_contour." + suf))))


def create_geojson(res_box, res_label, res_weight, image_path, start_time):
    image_name = image_path.split('/')[-1]
    out_name = image_name.split('.')[0] + '.geojson'
    out_path = image_path.replace(image_name, 'out/' + out_name)
    out_file = open(out_path, 'w', encoding='utf8')

    gdal.AllRegister()
    dataset = gdal.Open(image_path)
    adfGeoTransform = dataset.GetGeoTransform()

    res_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": []
    }
    time_now = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for index in range(len(res_box)):
        box = res_box[index]
        label = res_label[index]
        weight = res_weight[index]
        name = ''
        if label == 0:
            name = '风机'
        feature = {"type": "Feature",
                   "properties": {"Id": 0, "name": name, "date": time_now, "area": 0.0, "label": label,
                                  "weight": weight, "备注": 0},
                   "geometry": {"type": "Polygon", "coordinates": []}}
        coordinate = []
        for xy in box:
            location = [xy[0] * adfGeoTransform[1] + adfGeoTransform[0],
                        xy[1] * adfGeoTransform[5] + adfGeoTransform[3]]
            coordinate.append(location)
        coordinate.append(coordinate[0])
        feature['geometry']['coordinates'].append(coordinate)
        res_dict['features'].append(feature)
    end_time = time.time()
    consume_time = end_time - start_time
    res_dict['consume_time'] = consume_time
    print('图片路径：'+image_path + ' 预测耗时（单位s）：', consume_time)

    out_file.write(str(res_dict).replace('\'', '"').replace('None', '"None"'))
    out_file.close()


def single_predict(weights, image_path, split_arr):
    # 图像分割
    split_out_path = split_image(image_path, split_arr)
    # 预测
    all_box_arr, weight_arr, label_arr, image_paths = inference(weights, split_out_path)
    # 坐标变换
    all_box_arr, weight_arr, label_arr = predict_thread_run(all_box_arr, image_paths, label_arr, weight_arr,
                                                            split_arr)

    # 删除图像分割文件
    # shutil.rmtree(split_out_path)

    # 锚框去重
    flags = de_results(all_box_arr, weight_arr)
    res_box = []
    res_weight = []
    res_label = []
    for index in range(len(flags)):
        if flags[index] == 1:
            continue
        elif flags[index] == 0:
            res_box.append(all_box_arr[index])
            res_weight.append(weight_arr[index])
            res_label.append(label_arr[index])

    print('预测已完成')
    # res_box = np.array(res_box)
    # res_weight = np.array(res_weight)
    # res_label = np.array(res_label)
    # np.save('all_box_arr.npy', res_box)
    # np.save('weight_arr.npy', res_weight)
    # np.save('label_arr.npy', res_label)
    # print('预测数据已保存')

    # 生成带锚框的预测图片
    show_one_image(image_path, res_box, res_weight, res_label)

    # 保存锚框为geojson文件
    create_geojson(res_box, res_label, res_weight, image_path, start_time)

    # return res_box, res_weight, res_label

def batch_predict(image_file_path, weights, split_arr):
    # 读取配置文件
    image_names = os.listdir(image_file_path)
    for image_name in image_names:
        if '.' not in image_name:
            continue
        if '.json' in image_name or '.geojson' in image_name:
            continue
        # if image_name != 'car20210205.tif':
        #    continue
        print('预测，', image_name)
        image_path = image_file_path + '/' + image_name
        single_predict(weights, image_path, split_arr)

# 批量预测已经分割好的图片
def batch_predict_split_image(image_file_path, weights):
    # 读取配置文件
    # image_names = os.listdir(image_file_path)
    # 图像分割
    # 预测
    image_names = os.listdir(image_file_path)
    # output_path = image_file_path + '/out'
    # if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(output_path)
    # source_dir = '/'.join(image_file_path.split('/')[:-1])
    for image in image_names:
        if '.' not in image:
            continue
        print(image)
        image_path = image_file_path + '/' + image
        all_box_arr, weight_arr, label_arr, image_paths = inference(weights, image_path)
        flags = de_results(all_box_arr, weight_arr)
        res_box = []
        res_weight = []
        res_label = []
        for index in range(len(flags)):
            if flags[index] == 1:
                continue
            elif flags[index] == 0:
                res_box.append(all_box_arr[index])
                res_weight.append(weight_arr[index])
                res_label.append(label_arr[index])
        show_path = image_file_path + '/out/'
        # 大图打上标签
        if not os.path.exists(show_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(show_path)
        image_name = image_path.split('/')[-1]
        suf = image_name.split('.')[1]
        img = cv2.imread(image_path)
        box_len = len(res_box)
        colors = [random.randint(0, 255) for _ in range(3)]
        for i in range(box_len):
            label = res_label[i]
            confidence = res_weight[i]
            xyxy = [res_box[i][0][0], res_box[i][0][1], res_box[i][2][0], res_box[i][2][1]]
            plot_one_box(xyxy, img, label=f'{label} {confidence}', color=colors, line_thickness=2)
        cv2.imwrite(os.path.join(show_path, image_name.replace('.' + suf, "_contour." + suf)), img)
        print('已生成图片保存至{}'.format(os.path.join(show_path, image_name.replace('.' + suf, "_contour." + suf))))
    # 坐标变换
    # all_box_arr, weight_arr, label_arr = predict_thread_run(all_box_arr, image_paths, label_arr, weight_arr,
    #                                                         split_arr)

    # 删除图像分割文件
    # shutil.rmtree(split_out_path)

    # 锚框去重
    flags = de_results(all_box_arr, weight_arr)
    res_box = []
    res_weight = []
    res_label = []
    for index in range(len(flags)):
        if flags[index] == 1:
            continue
        elif flags[index] == 0:
            res_box.append(all_box_arr[index])
            res_weight.append(weight_arr[index])
            res_label.append(label_arr[index])

    print('预测已完成')
    # res_box = np.array(res_box)
    # res_weight = np.array(res_weight)
    # res_label = np.array(res_label)
    # np.save('all_box_arr.npy', res_box)
    # np.save('weight_arr.npy', res_weight)
    # np.save('label_arr.npy', res_label)
    # print('预测数据已保存')

    # 生成带锚框的预测图片
    show_one_image(image_path, res_box, res_weight, res_label)

    # 保存锚框为geojson文件
    create_geojson(res_box, res_label, res_weight, image_path, start_time)

    # return res_box, res_weight, res_label
if __name__ == '__main__':
    start_time = time.time()
    weights = "fan_models/fan_best_20230904.pt"
    image_path = "inference/20221119.tif"
    image_folder_path = "inference/Satellite_Images"

    split_arr = [[1300, 1300], [2000, 2000], [3000, 3000], [4000, 4000]]

    # single_predict(weights, image_path, split_arr)
    batch_predict(image_folder_path, weights, split_arr)
    # image_file_path = "inference/Test_Images/fan"
    # batch_predict_split_image(image_file_path, weights)

