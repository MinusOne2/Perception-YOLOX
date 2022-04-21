import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 参考链接：https://zhuanlan.zhihu.com/p/371743535

#1.标签路径
labelme_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/origin/Label/"    
img_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/VOC2007/JPEGImages/"            #使用labelme打的标签（包含每一张照片和对应json格式的标签）
saved_path = "C:/Users/oywt/Project/Pception-YOLOX/datasets/auto_pilot/VOC2007/"                #保存路径


#2.voc格式的文件夹，如果没有，就创建一个
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
    
#3.获取json文件
files = glob(labelme_path + "*.json")
files = [i.split("/")[-1].split("\\")[-1].split(".json")[0] for i in files]  #获取每一个json文件名


#4.读取每一张照片和对应标签，生成xml
num1 = 0
num2 = 0
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename,"r",encoding="utf-8"))
    height, width, channels = cv2.imread(img_path + json_file_ +".png").shape
    with codecs.open(saved_path + "Annotations/"+json_file_ + ".xml","w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'JPEGImages' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".png" + '</filename>\n')
        xml.write('\t<path>' + 'JPEGImages' + '</path>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["annotations"]:
            bbox = multi['bbox']
            xmin = int(bbox[0] - .5*bbox[2])
            xmax = int(bbox[0] + .5*bbox[2])
            ymin = int(bbox[1] - .5*bbox[3])
            ymax = int(bbox[1] + .5*bbox[3])

            category_id = multi['category_id']
            occlusion = multi['occlusion']
            truncation = multi['truncation']
            ignore = multi['ignore']

            # pedestrian、non-motor-vehicle、motor-vehicle、other
            if str(category_id) == 'other' or str(occlusion) == '3' or str(truncation) == '3' :
                num1 += 1 
                continue
            else:
                if str(category_id) == 'motor-vehicle':
                    if max(ymax-ymin,xmax-xmax) < 10:
                        num2 += 1 
                        continue
                
                elif str(category_id) == 'non-motor-vehicle':
                    if max(ymax-ymin,xmax-xmax) < 15:
                        num2 += 1 
                        continue
                
                else:
                    if max(ymax-ymin,xmax-xmax) < 30:
                        num2 += 1 
                        continue
                

                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + str(category_id) + '</name>\n')
                xml.write('\t\t<pose>' + 'Unspecified' + '</pose>\n')
                xml.write('\t\t<difficult>' + str(occlusion) + '</difficult>\n')
                xml.write('\t\t<truncated>' + str(truncation) + '</truncated>\n')
                xml.write('\t\t<ignore>' + str(ignore) + '</ignore>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename,xmin,ymin,xmax,ymax)
        xml.write('</annotation>')

print('num1:', num1)
print('num2:', num2)
