# 对head文件夹进行拆分，分为train.txt和val.txt
import os
import random

images_path = "C:/Users/oywt/Project/YOLOX/datasets/auto_pilot/Image/"
xmls_path = "C:/Users/oywt/Project/YOLOX/datasets/auto_pilot/annotation/"
train_val_txt_path = "C:/Users/oywt/Project/YOLOX/datasets/auto_pilot/ImageSets/"
val_percent = 0.3

'''
raw:   0 1 2 3 4 5 6 7 8 9(10) 
train: 0 1 2 4 5 6 8 (7)  n n+1 n+2 n+4
val:   3 7 9(3)           n+3
'''

images_list = os.listdir(images_path)
# random.shuffle(images_list)

#　划分训练集和验证集的数量
total_images_count = len(images_list)
train_images_count = int((1-val_percent)*len(images_list))
val_images_count = int(val_percent*len(images_list))

#　生成训练集的train.txt文件
train_txt = open(os.path.join(train_val_txt_path,"train.txt"),"w")
val_txt = open(os.path.join(train_val_txt_path,"val.txt"),"w")
train_count = 0
val_count = 0
for i in range(total_images_count):
    if i % 3 ==0 and i != 0:
        text = images_list[i].split(".png")[0] + "\n"
        val_txt.write(text)
        val_count+=1
        print("val_count: " + str(val_count))
    else:
        text = images_list[i].split(".png")[0] + "\n"
        train_txt.write(text)
        train_count+=1
        print("train_count: " + str(train_count))
train_txt.close()
val_txt.close()

# #　生成验证集的val.txt文件
# val_txt = open(os.path.join(train_val_txt_path,"val.txt"),"w")
# val_count = 0
# for i in range(val_images_count):
#     text = images_list[train_images_count + i].split(".jpg")[0] + "\n"
#     val_txt.write(text)
#     val_count+=1
#     print("val_count: " + str(val_count))
# val_txt.close()




