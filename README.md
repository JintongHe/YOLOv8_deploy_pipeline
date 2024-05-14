
# 部署YOLOv8到服务器的代码

## 1.环境安装  

> conda create --name myenv --file environment.yml

## 2.目录介绍

### 2.1 data_handle

用于存储预测中可重复使用的代码模块及函数

### 2.2 highway

高速公路项目识别代码

### 2.3 real_estate_v1  

通过影子预测建筑物高度的相关代码

### 2.4 server

预测接口相关代码。运行api.py启动后端接口

### 2.5 ultralytics_dev_v1

YOLOv8预测相关代码，包含房地产，风机，水泥厂等项目

### 2.6 yolosam_iron_chemical/yolosam_powerstation

火力发电及石化厂预测相关代码

### 2.7 yolov7

YOLOv7相关代码
