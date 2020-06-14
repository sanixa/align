#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
import scipy.io

image_size = 128          # 指定图片大小
path = '/home/uscc/Downloads/CUB_200_2011/CUB_200_2011/'   #文件读取路径

'''
classname = pd.read_csv(path+'classes.txt',header=None,sep = ' ')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字
'''

#根据目录读取一类图像，图片大小统一为image_size
def load_Img(imgDir):

    data = np.empty((image_size,image_size,3),dtype="float16")
    img = Image.open(imgDir)
    arr = np.asarray(img,dtype="float32")
    if arr.shape[1] > arr.shape[0]:
        arr = cv2.copyMakeBorder(arr,int((arr.shape[1]-arr.shape[0])/2),int((arr.shape[1]-arr.shape[0])/2),0,0,cv2.BORDER_CONSTANT,value=0)
    else:
        arr = cv2.copyMakeBorder(arr,0,0,int((arr.shape[0]-arr.shape[1])/2),int((arr.shape[0]-arr.shape[1])/2),cv2.BORDER_CONSTANT,value=0)       #长宽不一致时，用padding使长宽一致
    arr = cv2.resize(arr,(image_size,image_size))
    if len(arr.shape) == 2:
        temp = np.empty((image_size,image_size,3),dtype="float16")
        temp[:,:,0] = arr
        temp[:,:,1] = arr
        temp[:,:,2] = arr
        arr = temp
    if arr.shape == (image_size,image_size,4):
        temp = np.empty((image_size,image_size,3),dtype="float16")
        temp[:,:,0] = arr[:,:,0]
        temp[:,:,1] = arr[:,:,1]
        temp[:,:,2] = arr[:,:,2]
        arr = temp
    if arr.shape != (image_size,image_size,3):
        print('error', imgDir)
    return arr

def load_data():
    
    train_data_list = []
    train_label_list = []
    train_attr_list = []
    test_data_list = []
    test_label_list = []
    test_attr_list = []

    with open(path+'attributes/class_attribute_labels_continuous.txt', 'r') as f:
        attr = f.readlines()
    with open(path+'image_class_labels.txt', 'r') as f:
        label = f.readlines()
    with open(path+'images.txt', 'r') as f:
        images = f.readlines()
    with open(path+'train_test_split.txt', 'r') as f:
        split = f.readlines()
    
    for i in range(len(images)):#
        img_path = images[i].split()[1]
        img_label = str(int(label[i].split()[1]) -1) ####label 1~200 ==> 0~199

        tup = load_Img(path+ 'images/'+img_path)
        
        if split[i].split()[1] == str(1): ##train set
            train_data_list.append(tup)
            train_label_list += [img_label]
            train_attr_list += [attr[int(img_label)].split(' ')]
        elif split[i].split()[1] == str(0): ##test set
            test_data_list.append(tup)
            test_label_list += [img_label]
            test_attr_list += [attr[int(img_label)].split(' ')]
          
    return np.row_stack(train_data_list).reshape(-1, image_size, image_size, 3), np.array(train_label_list), np.array(train_attr_list), np.row_stack(test_data_list).reshape(-1, image_size, image_size, 3), np.array(test_label_list), np.array(test_attr_list)



traindata, trainlabel, trainattr, testdata, testlabel, testattr = load_data()
np.save(path+'traindata.npy',traindata)
np.save(path+'trainlabel.npy',trainlabel)
np.save(path+'trainattr.npy',trainattr)

print(traindata.shape,trainlabel.shape, trainattr.shape)

np.save(path+'testdata.npy',testdata)
np.save(path+'testlabel.npy',testlabel)
np.save(path+'testattr.npy',testattr)

print(testdata.shape,testlabel.shape, testattr.shape)




