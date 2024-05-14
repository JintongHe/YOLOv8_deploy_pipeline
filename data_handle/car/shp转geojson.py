# _*_ coding:utf-8 _*_
import geopandas as gpd
import os
import zipfile


def batch_read_shapfile():
    """
    遍历根目录读取shapefile文件
    :return:
    """
    # shapefile数据源根目录
    source_dir = input("===================== 请输入shapefile文件根目录,如E:\work ==========================\n")
    if not os.path.exists(source_dir):
        print("您输入的目录错误或者不存在!!!")
        source_dir = input("请重新输入:\n")

    # GeoJSON文件存储路径
    target_dir = os.path.join(source_dir, 'geojson')
    if not os.path.exists(target_dir):  # 如果target_dir目录不存在，则创建该目录
        os.makedirs(target_dir)

    shpfile_data = []
    # 1、批量读取shapefile文件
    for root_folder, sub_forder, filenames in os.walk(source_dir):
        for filename in filenames:
            shpfile = os.path.join(root_folder, filename)
            shpfile_data.append(shpfile)

    # 2、将shapfile文件批量转为geojson文件
    for shpfile in shpfile_data:
        if str(shpfile).endswith(".shp"):
            target_filename = os.path.basename(shpfile).split(".")[0]
            data = gpd.read_file(shpfile)
            geojson_file = os.path.join(target_dir, target_filename + ".geojson")

            data.crs = 'EPSG:4326'
            data.to_file(geojson_file, driver="GeoJSON")
            print("文件存储路径为：" + geojson_file)


def read_zip_file():
    """
    读取zip压缩文件
    :return:
    """
    # zip文件路径
    zip_filename = input("=====================  请输入zip文件路径,如E:\work\c.zip  ==========================\n")
    if not os.path.exists(zip_filename):
        print("您输入的目录错误或者不存在!!!")
        zip_filename = input("请重新输入:\n")

    # zip_filename = r'E:\gisdata\project\daxing\shpdata\shpdata.zip'
    # geojson文件存储路径
    zip_filename_path = os.path.dirname(zip_filename)
    shpfile_data = []
    # 1、从zip压缩文件中批量读取shapefile文件
    if zipfile.is_zipfile(zip_filename):  # 判断zip_filename是不是一个zip压缩文件
        f = zipfile.ZipFile(zip_filename)
        names = f.namelist()
        for name in names:
            if str(name).endswith('.shp'):
                shpfile_data.append(name)

    # 2、将shapfile文件批量转为geojson文件
    for shpfile in shpfile_data:
        data = gpd.read_file(os.path.join(zip_filename_path, shpfile))
        zip_target_filename = shpfile.split(".")[0]
        geojson_file = os.path.join(zip_filename_path, zip_target_filename + ".geojson")

        data.crs = 'EPSG:4326'
        data.to_file(geojson_file, driver="GeoJSON")
        print("文件存储路径为：" + geojson_file)


if __name__ == '__main__':
    file_type = input("=======================请选择数据源类型：1 文件夹，2 zip压缩包====================\n")
    if int(file_type) == 1:
        batch_read_shapfile()
    elif int(file_type) == 2:
        read_zip_file()
