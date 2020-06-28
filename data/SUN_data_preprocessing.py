#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
import scipy.io

image_size = 128          # 指定图片大小
path = '/home/uscc/Downloads/SUNAttributeDB/'   #文件读取路径


classname = pd.read_csv(path+'classes.txt',header=None,sep = ' ')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字
'''
images = scipy.io.loadmat(path +'images.mat')
attributes = scipy.io.loadmat(path +'attributeLabels_continuous.mat')
#print(images['images'][1][0][0])

img = Image.open('/Users/keiko/tensorflow/SUNAttributeDB/images/t/terrace_farm/sun_bmmnpaheooapqpes.jpg')
arr = np.asarray(img,dtype="float32")
print(arr[:,:,1])

def make_classes_txt():
    classes = os.listdir(path+ 'images')
    classname = []
    cnt =0
    for i in range(len(classes)):
        temp = os.listdir(path+ 'images/' + classes[i])
        for item in temp:
            classname.append(str(cnt) + ',' + classes[i] + '/' + item + '\n')
            cnt+=1
        
    with open(path+'classes.txt', 'w') as f:
        f.writelines(classname)
        
#make_classes_txt()
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

def load_data(data):
    
    data_list = []
    label_list = []
    attr_list = []
   
    
    for i in range(len(images['images'])):#
        split_class_idx = images['images'][i][0][0].find('/', 2)
        if images['images'][i][0][0][:split_class_idx] + '\n' not in data:
            continue
        tup = load_Img(path+'images/'+images['images'][i][0][0])
        data_list.append(tup)

        label_list += [dic_name2class[images['images'][i][0][0][:split_class_idx]]]

        attr_list += [attributes['labels_cv'][i]]
          
    return np.row_stack(data_list).reshape(-1, image_size, image_size, 3),np.array(label_list), np.array(attr_list)


#train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
#test_classes = pd.read_csv(path+'testclasses.txt',header=None)

trainclasses = []
testclasses = []
with open(path+'trainclasses.txt', 'r') as f:
    trainclasses = f.readlines()
with open(path+'testclasses.txt', 'r') as f:
    testclasses = f.readlines()

traindata,trainlabel,trainattr = load_data(trainclasses)

np.save(path+'traindata.npy',traindata)
np.save(path+'trainlabel.npy',trainlabel)
np.save(path+'trainattr.npy',trainattr)

print(traindata.shape,trainlabel.shape, trainattr.shape)

testdata,testlabel,testattr = load_data(testclasses)
np.save(path+'testdata.npy',testdata)
np.save(path+'testlabel.npy',testlabel)
np.save(path+'testattr.npy',testattr)

print(testdata.shape,testlabel.shape, testattr.shape)

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
        if arr.shape == (image_size,image_size,4):
            temp = np.empty((image_size,image_size,3),dtype="float16")
            temp[:,:,0] = arr[:,:,0]
            temp[:,:,1] = arr[:,:,1]
            temp[:,:,2] = arr[:,:,2]
            arr = temp
        if arr.shape != (image_size,image_size,3):
            print('error', imgDir)
        data[i,:,:,:] = arr
    return data,imgNum 

def load_data(data, num):
    read_num = num
    
    data_list = []
    label_list = []
    attr_list = []

    matcontent = scipy.io.loadmat('SUN/att_splits.mat')
    attr = matcontent['att'].T
    for i in range(len(attr)):
        for j in range(len(attr[0])):
            attr[i,j] = str(attr[i,j])
    
    for item in data:#.iloc[:,0].values.tolist()
        item = item[:-1] ##remove '\n'
        tup = load_Img(path+'images/'+ item[0] + '/' +item,read_num=read_num)
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
