import cv2
import time
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

def load_resized_img(self, index):
    time1 = time.time()
    img = self.load_image(index)
    time2 = time.time()
    print("loading time:", time2 - time1)
    r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])

    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    time3 = time.time()
    print("resize time:", time3 - time2)

    return resized_img

def load_image(self, index):
    img_id = self.ids[index]
    img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
    # img = mx.image.imdecode(open(self._imgpath % img_id,'rb').read())
    assert img is not None

    return img

#1.标签路径
labelme_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/origin/Label/"    
img_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/VOC2007/JPEGImages/"            #使用labelme打的标签（包含每一张照片和对应json格式的标签）
saved_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/VOC2007/ResizedJPEGImages/"                #保存路径


    
#3.获取json文件
files = glob(img_path + "*.png")
files = [i.split("/")[-1].split("\\")[-1].split(".png")[0] for i in files]  #获取每一个json文件名


for json_file_ in files:
    
    time1 = time.time()
    img = cv2.imread(img_path + json_file_ +".png")
    time2 = time.time()
    # print("loading time:", time2 - time1)
    r = min(384 / img.shape[0], 640 / img.shape[1])

    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    time3 = time.time()
    # print("resize time:", time3 - time2)
    cv2.imwrite(saved_path + json_file_ +".png", resized_img, [int(cv2.IMWRITE_PNG_COMPRESSION),3])
    print(saved_path + json_file_ + ".png")
