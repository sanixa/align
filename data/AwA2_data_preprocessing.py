#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image

image_size = 224          # 指定图片大小
path = '/home/uscc/Downloads/Animals_with_Attributes2/'   #文件读取路径

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字

#根据目录读取一类图像，read_num指定每一类读取多少图片，图片大小统一为image_size
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
   
    
    for item in data.iloc[:,0].values.tolist():
        tup = load_Img(path+'JPEGImages/'+item,read_num=read_num)
        data_list.append(tup[0])
        label_list += [dic_name2class[item]]*tup[1]
        
          
    
    return np.row_stack(data_list),np.array(label_list)

train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
test_classes = pd.read_csv(path+'testclasses.txt',header=None)

traindata,trainlabel = load_data(train_classes, num='max')
np.save(path+'AwA2_seen_x.npy',traindata)
np.save(path+'AwA2_seen_y.npy',trainlabel)

print(traindata.shape,trainlabel.shape)

testdata,testlabel = load_data(test_classes, num='max')
np.save(path+'AwA2_unseen_x.npy',testdata)
np.save(path+'AwA2_unseen_y.npy',testlabel)

print(testdata.shape,testlabel.shape)

