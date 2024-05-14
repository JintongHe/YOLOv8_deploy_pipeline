# coding = utf8
import os
from shutil import copyfile
import random
import os
from tqdm import tqdm
import cv2
import os
import shutil


def imgFlipping(path):
    # path = 'E:/work/python/ai/fan/datasets'
    image_names = os.listdir(path + '/images')
    image_labels = os.listdir(path + '/labels')
    print(len(image_names))
    print(len(image_labels))

    out_image_path = path + '/images_flip'
    out_label_path = path + '/labels_flip'
    folder = os.path.exists(out_image_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_image_path)
    folder = os.path.exists(out_label_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_label_path)

    for image_name in image_names:
        prefix = image_name.split('.')[0]
        suffix = image_name.split('.')[1]

        s = os.path.join(path + '/images', image_name)
        tmp = cv2.imread(s)
        # 翻转图像
        imgFlip0 = cv2.flip(tmp, 0)  # 上下
        imgFlip1 = cv2.flip(tmp, 1)  # 左右
        imgFlip2 = cv2.flip(tmp, -1)  # 上下左右
        cv2.imwrite(os.path.join(out_image_path, prefix + "_0." + suffix), imgFlip0)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_1." + suffix), imgFlip1)
        cv2.imwrite(os.path.join(out_image_path, prefix + "_2." + suffix), imgFlip2)

        if prefix + '.txt' in image_labels:
            # f = open(os.path.join(files, filesDir[i].replace(".bmp",".txt")),"r")
            f = open(os.path.join(path + '/labels', prefix + '.txt'), "r")
            lines = f.readlines()
            f.close()
            # 以下为YOLO目标检测格式转换
            # 上下翻转，Y坐标变化为1-Y
            tmp0 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                num = 2
                tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp0 += tmpK

            f = open(os.path.join(out_label_path, prefix + "_0.txt"), "w")
            f.writelines(tmp0, )
            f.close()

            # 左右翻转，X坐标变化为1-X
            tmp1 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                num = 1
                tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp1 += tmpK
            f = open(os.path.join(out_label_path, prefix + "_1.txt"), "w")
            f.writelines(tmp1, )
            f.close()

            # 上下左右翻转，X坐标变化为1-X，Y坐标变化为1-Y
            tmp2 = ""
            for line in lines:
                tmpK = line.strip().split(' ')
                num = 1
                tmpK[num] = str(1 - float(tmpK[num]))
                num = 2
                tmpK[num] = str(1 - float(tmpK[num]))
                tmpK = (" ".join(tmpK) + "\n")
                # print(tmp2)
                tmp2 += tmpK
            f = open(os.path.join(out_label_path, prefix + "_2.txt"), "w")
            f.writelines(tmp2, )
            f.close()


# 合并数据
def merge_data(path):
    # path = 'E:/work/python/ai/fan/datasets'
    target1 = path + '\\images_all'
    target2 = path + '\\labels_all'
    # 源1
    image_names = os.listdir(path + '\\images')
    image_labels = os.listdir(path + '\\labels')

    for name in image_names:
        source = path + '\\images\\' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = path + '\\labels\\' + label
        copyfile(source, target2 + '/' + label)

    # 源2
    image_names = os.listdir(path + '\\images_flip')
    image_labels = os.listdir(path + '\\labels_flip')
    for name in image_names:
        source = path + '\\images_flip\\' + name
        copyfile(source, target1 + '/' + name)

    for label in image_labels:
        source = path + '\\labels_flip\\' + label
        copyfile(source, target2 + '/' + label)



# 生成yoyov7所需的train、test、val。path代表存images和labels的地方
def get_train_test_val(path):
    # path = 'E:\\work\\python\\ai\\fan\\datasets'
    image_names = os.listdir(path + '\\images_all')
    image_labels = os.listdir(path + '\\labels_all')

    print(len(image_names))
    print(len(image_labels))
    # 创建相关文件夹
    for name in ['train', 'test', 'val']:
        if os.path.exists(path + '\\' + name):
            shutil.rmtree(path + '\\' + name)  # 递归删除文件夹，即：删除非空文件夹
        os.makedirs(path + '\\' + name)
        os.makedirs(path + '\\' + name + '\\images')
        os.makedirs(path + '\\' + name + '\\labels')

    sums = len(image_names)
    num = 0
    for name in image_names:
        source = path + '\\images_all\\' + name
        num += 1
        if num % 1000 == 0:
            print(num)

        rand_num = random.randint(1, sums)
        rand_rate = rand_num / sums

        if rand_rate <= 0.7:
            target = path + '\\train\\images\\' + name
        elif rand_rate <= 0.9:
            target = path + '\\test\\images\\' + name
        else:
            target = path + '\\val\\images\\' + name
        copyfile(source, target)

        if name.replace('.jpg', '.txt') in image_labels:
            source = path + '\\labels_all\\' + name.replace('.jpg', '.txt')
            if rand_rate <= 0.7:
                target = path + '\\train\\labels\\' + name.replace('.jpg', '.txt')
            elif rand_rate <= 0.9:
                target = path + '\\test\\labels\\' + name.replace('.jpg', '.txt')
            else:
                target = path + '\\val\\labels\\' + name.replace('.jpg', '.txt')

            copyfile(source, target)


# 生成train、test、val,txt文件存图片路径。
# path1代表txt文件所在地址，txt文件包含train.txt、test.txt、val.txt
# path2代表images所在地址，包含train、test、val三种情况
def get_train_test_val_txt(path1, path2):
    for name in ['train', 'test', 'val']:
        # path1 = 'E:/work/python/ai/fan/yolov7/datasets'
        # path2 = 'E:/work/python/ai/fan/datasets'
        f = open(path1 + '/' + name + '.txt', 'w', encoding='utf8')
        path = path2 + '/' + name + '/images'
        image_names = os.listdir(path)
        for image_name in image_names:
            f.write(path + '/' + image_name + '\n')
        f.close()


if __name__ == "__main__":
    path1 = 'E:/work/python/ai/fan/yolov7/datasets'
    path2 = 'E:/work/python/ai/fan/datasets'
    get_train_test_val_txt(path1, path2)

