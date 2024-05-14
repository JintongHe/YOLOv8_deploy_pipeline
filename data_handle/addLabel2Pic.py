import cv2
import os
import numpy as np
from tqdm import tqdm


# 在图片中显示数据
def label2pic(images_path, labels_path, images_show_path, class_label):
    files = os.listdir(images_path)
    img_format = '.png'
    img_list = [x for x in files if x.endswith(img_format)]

    for item in tqdm(img_list):
        img = cv2.imread(os.path.join(images_path, item))
        w = img.shape[1]
        h = img.shape[0]

        contours = []
        if not os.path.exists(os.path.join(labels_path, item.replace(img_format, ".txt"))):
            continue
        with open(os.path.join(labels_path, item.replace(img_format, ".txt")), "r") as t:
            s_label = []
            f = t.readlines()
            for line in f:
                arr = line.strip().split(" ")
                contours.append(arr[1:])
                s_label.append(arr[0])

        contours_tmp = []
        length_max = 0

        label_position = []

        for contour in contours:
            length = len(contour)
            if int(length / 2) > length_max:
                length_max = int(length / 2)

            tmp_position = []
            contour_tmp = []
            for i in range(len(contour)):
                if i % 2 == 0:
                    tmp = list()
                    tmp_x = int(float(contour[i]) * w)
                    tmp_y = int(float(contour[i + 1]) * h)
                    tmp.append([tmp_x, tmp_y])
                    contour_tmp.append(tmp)

                    if len(tmp_position) == 0:
                        tmp_position = [tmp_x, tmp_y]
                    else:
                        if tmp_position[0] > tmp_x:
                            tmp_position[0] = tmp_x
                            tmp_position[1] = tmp_y
                        else:
                            pass
            label_position.append(tuple(tmp_position))
            contours_tmp.append(contour_tmp)

        for i in range(len(contours_tmp)):
            if len(contours_tmp[i]) < length_max:
                s = contours_tmp[i][-1]
                for j in range(length_max - len(contours_tmp[i])):
                    contours_tmp[i].append(s)
            contours_tmp[i] = np.array(contours_tmp[i])

        contours_tmp = tuple(contours_tmp)

        img_contour = cv2.drawContours(img, contours_tmp, -1, (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        print(label_position)
        print(len(label_position))
        print(contours_tmp)
        print(len(contours_tmp))
        for num in range(len(label_position)):
            img_contour = cv2.putText(img_contour, class_label[int(s_label[num])], label_position[num], font, 0.4, color, 1)
        cv2.imwrite(os.path.join(images_show_path, item.replace(img_format, "_contour" + img_format)), img_contour)


def demo():
    images_path = r"E:\work\python\ai\data_handle\car\test"
    labels_path = r"E:\work\python\ai\data_handle\car\test"
    images_show_path = r"E:\work\python\ai\data_handle\car\test\show"
    class_label = ['build', 'car']
    label2pic(images_path, labels_path, images_show_path, class_label)


if __name__ == '__main__':
    demo()
