#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
import scipy.io

image_size = 128          # 指定图片大小
path = '/home/uscc/Downloads/CUB_200_2011/CUB_200_2011/'   #文件读取路径


classname = pd.read_csv(path+'classes.txt',header=None,sep = ' ')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字

#根据目录读取一类图像，图片大小统一为image_size
'''
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

    #with open(path+'attributes/class_attribute_labels_continuous.txt', 'r') as f:
       #attr = f.readlines()
    matcontent = scipy.io.loadmat('CUB/att_splits.mat')
    attr = matcontent['att'].T
    with open(path+'image_class_labels.txt', 'r') as f:
        label = f.readlines()
    with open(path+'images.txt', 'r') as f:
        images = f.readlines()

    
    for i in range(len(images)):#
        img_path = images[i].split()[1]
        img_label = str(int(label[i].split()[1]) -1) ####label 1~200 ==> 0~199

        tup = load_Img(path+ 'images/'+img_path)
        
        if int(img_label) <= 160: ##train set
            train_data_list.append(tup)
            train_label_list += [img_label]
            train_attr_list += [attr[int(img_label)]]
        elif int(img_label) > 160: ##test set
            test_data_list.append(tup)
            test_label_list += [img_label]
            test_attr_list += [attr[int(img_label)]]
          
    return np.row_stack(train_data_list).reshape(-1, image_size, image_size, 3), np.array(train_label_list), np.array(train_attr_list), np.row_stack(test_data_list).reshape(-1, image_size, image_size, 3), np.array(test_label_list), np.array(test_attr_list)

'''

def load_Img(imgDir,read_num = 'max'):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    if read_num == 'max':
        imgNum = len(imgs)
    else:
        imgNum = read_num
    data = np.empty((imgNum,image_size,image_size,3),dtype="float16")
    print(imgNum)
    for i in range (imgNum):
        img = Image.open(imgDir+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        if arr.shape[1] > arr.shape[0]:
            arr = cv2.copyMakeBorder(arr,int((arr.shape[1]-arr.shape[0])/2),int((arr.shape[1]-arr.shape[0])/2),0,0,cv2.BORDER_CONSTANT,value=0)
        else:
            arr = cv2.copyMakeBorder(arr,0,0,int((arr.shape[0]-arr.shape[1])/2),int((arr.shape[0]-arr.shape[1])/2),cv2.BORDER_CONSTANT,value=0)       #长宽不一致时，用padding使长宽一致
        arr = cv2.resize(arr,(image_size,image_size))
        if len(arr.shape) == 2:
            temp = np.empty((image_size,image_size,3))
            temp[:,:,0] = arr
            temp[:,:,1] = arr
            temp[:,:,2] = arr
            arr = temp        
        data[i,:,:,:] = arr
    return data,imgNum 

def load_data(data, num):
    read_num = num
    
    data_list = []
    label_list = []
    attr_list = []

    matcontent = scipy.io.loadmat('CUB/att_splits.mat')
    attr = matcontent['att'].T
    
    for item in data:#.iloc[:,0].values.tolist()
        item = item[:-1] ##remove '\n'
        tup = load_Img(path+'images/'+item,read_num=read_num)
        data_list.append(tup[0])
        label_list += [dic_name2class[item]]*tup[1]
        attr_list += [attr[dic_name2class[item]]]*tup[1]
          
    
    return np.row_stack(data_list),np.array(label_list), np.array(attr_list)

trainclasses = []
testclasses = []
with open(path+'trainclasses.txt', 'r') as f:
    trainclasses = f.readlines()
with open(path+'testclasses.txt', 'r') as f:
    testclasses = f.readlines()
    

traindata, trainlabel, trainattr = load_data(trainclasses, num='max')
np.save(path+'traindata.npy',traindata)
np.save(path+'trainlabel.npy',trainlabel)
np.save(path+'trainattr.npy',trainattr)

print(traindata.shape,trainlabel.shape, trainattr.shape)

testdata, testlabel, testattr = load_data(testclasses, num='max')
np.save(path+'testdata.npy',testdata)
np.save(path+'testlabel.npy',testlabel)
np.save(path+'testattr.npy',testattr)

print(testdata.shape,testlabel.shape, testattr.shape)



