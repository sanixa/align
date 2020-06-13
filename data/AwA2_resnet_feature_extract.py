#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np

import keras
from keras.datasets import cifar10
from keras_applications.resnet import ResNet101



path = '/Users/keiko/tensorflow/Animals_with_Attributes2/'   #文件读取路径
image_size = 224

###################load data-----------------------------------------------------
traindata = np.load(path+'AWA2_traindata_100.npy')
#trainlabel = np.load(path+'AWA2_trainlabel.npy')

testdata = np.load(path+'AWA2_testdata_100.npy')
#testlabel = np.load(path+'AWA2_testlabel.npy')


#############normalize--------------------------------------------------------
traindata = traindata.astype('float16') / 255.
testdata = testdata.astype('float16') / 255.



model = ResNet101(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3), backend=keras.backend,
                  layers=keras.layers, models=keras.models, utils=keras.utils, pooling='avg')

traindata = model.predict(traindata)
np.save(path+'AWA2_resnet101_traindata_100.npy',traindata)

testdata = model.predict(testdata)
np.save(path+'AWA2_resnet101_testdata_100.npy',testdata)
