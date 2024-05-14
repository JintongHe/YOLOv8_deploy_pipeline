import cv2
import os
import shutil
from tqdm import tqdm

imgFiles = r"D:\solarpanelData2"
imgList = os.listdir(imgFiles)
imgFormat = ".jpg"
imgList = [x for x in imgList if x.endswith(imgFormat)]

for item in tqdm(imgList):
    file = os.path.join(imgFiles, item)
    img = cv2.imread(file)
    # # ---------------------------------
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_hsv[:,:,0] = img_hsv[:,:,0] * 0.85
    # img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite(file.replace(imgFormat, "_0_"+imgFormat), img1)
    # shutil.copy(file.replace(imgFormat, ".txt"), file.replace(imgFormat, ".txt").replace(".txt", "_0_.txt"))
    # ---------------------------------
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,1] = img_hsv[:,:,1] * 0.3
    img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(file.replace(imgFormat, "_1_"+imgFormat), img2)
    shutil.copy(file.replace(imgFormat, ".txt"), file.replace(imgFormat, ".txt").replace(".txt", "_1_.txt"))
    # ---------------------------------
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = img_hsv[:,:,2] * 0.5
    img_hsv[:,:,1] = img_hsv[:,:,1] * 0.3
    img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(file.replace(imgFormat, "_2_"+imgFormat), img1)
    shutil.copy(file.replace(imgFormat, ".txt"), file.replace(imgFormat, ".txt").replace(".txt", "_2_.txt"))
