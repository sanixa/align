#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
import csv
from PIL import Image

image_size = 224          # 指定图片大小
path = '/home/uscc/Downloads/Animals_with_Attributes2/' 

train_ratio = 0.8

seen_x = np.load(path + 'AwA2_seen_x.npy')
with open('AwA2_seen_x_train.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(int(len(seen_x) * train_ratio)):
    # 寫入一列資料
        writer.writerow(seen_x[i].reshape(image_size*image_size*3))
        
with open('AwA2_seen_x_test.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(int(len(seen_x) * train_ratio), len(seen_x)):
    # 寫入一列資料
        writer.writerow(seen_x[i].reshape(image_size*image_size*3))

seen_y = np.load(path + 'AwA2_seen_y.npy')
with open('AwA2_seen_y_train.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(int(len(seen_y) * train_ratio)):
    # 寫入一列資料
        writer.writerow([seen_y[i]])
        
with open('AwA2_seen_y_test.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(int(len(seen_y) * train_ratio), len(seen_y)):
    # 寫入一列資料
        writer.writerow([seen_y[i]])

#########---------------------------------------------------------------------
unseen_x = np.load(path + 'AwA2_unseen_x.npy')
#print(train_x[0].reshape(image_size*image_size*3))

with open('AwA2_unseen_x.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(unseen_x)):
    # 寫入一列資料
        writer.writerow(unseen_x[i].reshape(image_size*image_size*3))

unseen_y = np.load(path + 'AwA2_unseen_y.npy')
#print(train_y[0])
with open('AwA2_unseen_y.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(unseen_y)):
    # 寫入一列資料
        writer.writerow([unseen_y[i]])

'''
num_classes = 50
temp = np.eye(num_classes).reshape(num_classes*num_classes)

with open('AwA2_y_class_train.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(seen_x) * train_ratio):
    # 寫入一列資料
        writer.writerow(temp)
        
with open('AwA2_y_class_test.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(seen_x) * (1 - train_ratio)):
    # 寫入一列資料
        writer.writerow(temp)
'''
#y_class_train = np.repeat(a=temp, repeats=3200, axis=0).reshape(3200, num_classes, num_classes)
#y_class_test =  np.repeat(a=temp, repeats=800, axis=0).reshape(800, num_classes, num_classes)

