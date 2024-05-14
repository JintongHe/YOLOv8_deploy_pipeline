from osgeo import gdal
import os

import numpy as np


def write_image(filename, img_proj, img_geotrans, img_data):
    # 判断栅格数据类型
    if 'int8' in img_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判断数组维度
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
    else:
        img_bands, (img_height, img_width) = 1, img_data.shape

    # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    image = driver.Create(filename, img_width, img_height, img_bands, datatype)

    image.SetGeoTransform(img_geotrans)
    image.SetProjection(img_proj)

    if img_bands == 1:
        image.GetRasterBand(1).WriteArray(img_data)
    else:
        for i in range(img_bands):
            image.GetRasterBand(i + 1).WriteArray(img_data[i])

    del image  # 删除变量,保留数据


def divide_image(type_num, path, m, n, out):
    in_ds1 = gdal.Open(path)  # 读取原始图像文件信息
    xsize = in_ds1.RasterXSize
    ysize = in_ds1.RasterYSize
    bands = in_ds1.RasterCount
    geotransform = in_ds1.GetGeoTransform()
    projection = in_ds1.GetProjectionRef()
    data = in_ds1.ReadAsArray(0, 0, xsize, ysize)
    data1 = data * 0

    patch_ysize = int(ysize / n)
    patch_xsize = int(xsize / m)
    x_mod = xsize % m
    y_mod = ysize % n

    num = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            num += 1
            outfile = out + "\\mark" + str(type_num) + '_' + str(j) + '_' + str(i) + ".tif"
            if i == n and j == m:
                div_image = data[:, (i - 1) * patch_ysize: i * patch_ysize + y_mod,
                            (j - 1) * patch_xsize: j * patch_xsize + x_mod]
            elif i == n:
                div_image = data[:, (i - 1) * patch_ysize: i * patch_ysize + y_mod,
                            (j - 1) * patch_xsize: j * patch_xsize]
            elif j == m:
                div_image = data[:, (i - 1) * patch_ysize: i * patch_ysize,
                            (j - 1) * patch_xsize: j * patch_xsize + x_mod]
            else:
                div_image = data[:, (i - 1) * patch_ysize: i * patch_ysize, (j - 1) * patch_xsize: j * patch_xsize]
            write_image(outfile, projection, geotransform, div_image)
    return xsize, ysize, data1, projection, geotransform, x_mod, y_mod


def merge_image(m, n, xsize, ysize, data1, projection, geotransform, out, out1, x_mod, y_mod):
    patch_ysize = int(ysize / n)
    patch_xsize = int(xsize / m)
    print("m, n", m, n)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cut_image = out + "\\" + str(i) + '_' + str(j) + ".tif"
            in_ds1 = gdal.Open(cut_image)  # 读取原始图像文件信息
            xsize = in_ds1.RasterXSize
            ysize = in_ds1.RasterYSize
            data = in_ds1.ReadAsArray(0, 0, xsize, ysize)
            if i == n and j == m:
                data1[:, (i - 1) * patch_ysize: i * patch_ysize + y_mod,
                (j - 1) * patch_xsize: j * patch_xsize + x_mod] = data
            elif i == n:
                data1[:, (i - 1) * patch_ysize: i * patch_ysize + y_mod, (j - 1) * patch_xsize: j * patch_xsize] = data
            elif j == m:
                data1[:, (i - 1) * patch_ysize: i * patch_ysize, (j - 1) * patch_xsize: j * patch_xsize + x_mod] = data
            else:
                data1[:, (i - 1) * patch_ysize: i * patch_ysize, (j - 1) * patch_xsize: j * patch_xsize] = data

    outfile1 = out1 + '\\' + 'merge.tif'
    write_image(outfile1, projection, geotransform, data1)


def demo():
    in_ds = gdal.Open(r'E:\work\data\yushan\202104\202104.tif')  # 读取要切的原图
    print("open tif file succeed")
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    outbandsize = in_ds.RasterCount  # 获取数据波段数
    data = in_ds.ReadAsArray(0, 0, width, height)
    print(type(data))
    print(data.shape)
    print(type(data[0][0][0]))

    print("数据宽度", width)
    print("数据高度", height)
    print("波段数", outbandsize)
    in_ds = gdal.Open(r'E:\work\data\yushan\202104\joint\merge.tif')  # 读取要切的原图
    print("open tif file succeed")
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    outbandsize = in_ds.RasterCount  # 获取数据波段数
    data = in_ds.ReadAsArray(0, 0, width, height)
    print(type(data))
    print(data.shape)
    print(type(data[0][0][0]))

    print("数据宽度", width)
    print("数据高度", height)
    print("波段数", outbandsize)


def main1():
    path = r"E:\work\python\ai\data_handle\car\03230303.tif"
    out = r"E:\work\python\ai\data_handle\car\split"
    # m代表行的分块数,n代表列的分块数，m，n可以根据像素点个数和分割图像分辨率确定。这里可以改进代码
    in_ds = gdal.Open(path)  # 读取要切的原图
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    split_arr = [[1300, 1300], [800, 800], [500, 500], [1000, 700], [700, 1000]]
    for num in range(len(split_arr)):
        m = int(width / split_arr[num][0])
        n = int(height / split_arr[num][1])
        xsize, ysize, data1, projection, geotransform, x_mod, y_mod = divide_image(num, path, m, n, out)

    print('分割完成')


# 切分标注数据
def mark_split():
    source_dir = r'E:/work/data/car20230705/mark'
    file1_names = os.listdir(source_dir)
    for file1_name in file1_names:
        if '.' in file1_name:
            continue
        file2_names = os.listdir(source_dir + '/' + file1_name)
        for file2_name in file2_names:
            file3_names = os.listdir(source_dir + '/' + file1_name + '/' + file2_name)
            for file3_name in file3_names:
                if not file3_name.endswith('.tif'):
                    continue
                file3_name_replace = file3_name.replace(' ', '').replace('（', '(').replace('）', ')')
                if '(18).tif' in file3_name.replace(' ', '').replace('（', '(').replace('）', ')'):
                    continue
                # 级别
                level = file3_name_replace.split('(')[1].split(')')[0]
                path = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_name
                out = source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_name.replace('.tif', 'split')
                if not os.path.exists(out):
                    os.makedirs(out)
                # m代表行的分块数,n代表列的分块数
                in_ds = gdal.Open(path)  # 读取要切的原图
                width = in_ds.RasterXSize  # 获取数据宽度
                height = in_ds.RasterYSize  # 获取数据高度
                split_arr = [[1300, 1300], [800, 800], [500, 500], [1000, 700], [700, 1000]]
                sets = set()
                for num in range(len(split_arr)):
                    m = int(width / split_arr[num][0])
                    n = int(height / split_arr[num][1])
                    if m == 0:
                        m = 1
                    if n == 0:
                        n = 1
                    if (m, n) in sets:
                        continue
                    else:
                        sets.add((m, n))
                    print(width, height, split_arr[num], m, n)
                    divide_image(num, path, m, n, out)

                print(source_dir + '/' + file1_name + '/' + file2_name + '/' + file3_name, '分割完成')


if __name__ == '__main__':
    mark_split()
