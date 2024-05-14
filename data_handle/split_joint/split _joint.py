import cv2
import numpy as np
import os


def split():
    pic_path = '../data/20221119fan.tif'  # 分割的图片的位置
    pic_target = 'split_data/fan_split'  # 分割后的图片保存的文件夹
    if not os.path.exists(pic_target):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pic_target)
    # 要分割后的尺寸
    cut_width = 1024
    cut_length = 1024
    # 读取要分割的图片，以及其尺寸等数据
    picture = cv2.imread(pic_path)
    print(picture.shape)
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    # for循环迭代生成
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
            cv2.imwrite(result_path, pic)

    print("done!!!")


def joint():
    # 分割后的图片的文件夹，以及拼接后要保存的文件夹
    pic_path = 'split_joint/split_data/fan_split'
    pic_target = 'joint_data/'
    # 数组保存分割后图片的列数和行数，注意分割后图片的格式为x_x.jpg，x从1开始
    num_width_list = []
    num_lenght_list = []
    # 读取文件夹下所有图片的名称
    picture_names = os.listdir('split_data')
    if len(picture_names) == 0:
        print("没有文件")

    else:
        # 获取分割后图片的尺寸
        img_1_1 = cv2.imread(pic_path + '1_1.jpg')
        (width, length, depth) = img_1_1.shape
        # 分割名字获得行数和列数，通过数组保存分割后图片的列数和行数
        for picture_name in picture_names:
            num_width_list.append(int(picture_name.split("_")[-2].replace('split', '')))
            num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))
        # 取其中的最大值
        num_width = max(num_width_list)
        num_length = max(num_lenght_list)
        # 预生成拼接后的图片
        splicing_pic = np.zeros((num_width * width, num_length * length, depth))
        # 循环复制
        for i in range(1, num_width + 1):
            for j in range(1, num_length + 1):
                img_part = cv2.imread(pic_path + '{}_{}.jpg'.format(i, j))
                splicing_pic[width * (i - 1): width * i, length * (j - 1): length * j, :] = img_part
        # 保存图片，大功告成
        cv2.imwrite(pic_target + 'result.jpg', splicing_pic)
        print("done!!!")


if __name__ == '__main__':
    joint()

