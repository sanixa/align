#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras_applications.resnet import ResNet50
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


missing_class = [0]
batch_size = 100
original_dim = 3072
latent_dim = 128 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256
epochs = 1000
num_classes = 10
FREEZE_LAYERS = 176
unseen_class = [0]

'''
def split_dataset(args):
    xdata, ydata, uc = args
    index = []
    unseen_x = []
    unseen_y = []
    
    for i in range(len(ydata)):
        if(ydata[i] in uc):
            index.append(i)
            unseen_x.append(xdata[i])
            unseen_y.append(ydata[i])
            
    seen_x = np.delete(xdata, index, axis = 0)
    seen_y = np.delete(ydata, index)

    return seen_x, seen_y, np.row_stack(unseen_x), np.row_stack(unseen_y)


(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
seen_x, seen_y, unseen_x, unseen_y = split_dataset((x_train, y_train_, unseen_class))
unseen_x = unseen_x.reshape((5000, 32, 32, 3))

np.save('data/cifar10/seen_x.npy', seen_x)
np.save('data/cifar10/seen_y.npy', seen_y)
np.save('data/cifar10/unseen_x.npy', unseen_x)
np.save('data/cifar10/unseen_y.npy', unseen_y)
'''

'''
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)
x_train_nonflat = x_train
x_test_nonflat = x_test
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
'''


'''
ye = load_model('cnn_miss_0/y_encoder.h5')
ym = ye.predict(np.eye(num_classes - 1))

dot1 = np.dot(ym[0], ym[0])
dot2 = np.dot(ym[0], ym[0])
dot3 = np.dot(ym[0], ym[0])
print(dot1 / np.sqrt(dot2 * dot3))
'''

model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), backend=keras.backend,
                  layers=keras.layers, models=keras.models, utils=keras.utils, pooling='avg')
model.save('resnet50.h5')
'''
temp = np.eye(num_classes).reshape(1,100)
y_class = np.repeat(a=temp, repeats=50000, axis=0).reshape(50000,10,10)
print(y_class[0])
'''
'''
temp = np.eye(num_classes).reshape(1,100)
print(temp)
y_class = np.repeat(a=temp, repeats=50000, axis=0)
print(y_class.reshape(50000,10,10)[0])
'''
'''
y = Input(shape=(num_classes,)) # 输入类别
class_y = Input(shape=(10,))
yh = Dense(latent_dim)(y) # 这里就是直接构建每个类别的均值
y_encoder = Model(y, yh)
y_mean = y_encoder(class_y)
test = Model([class_y], y_mean)

a = test.predict(np.eye(num_classes))
print(a[0])
'''
'''
x = Input(shape=(original_dim,))
x2 = Input(shape=(original_dim,))

def parm_layer(args):
    m1, m2 = args
    return m1 - m2

diff = Lambda(parm_layer, output_shape=(latent_dim,))([x, x2])

vae = Model(inputs=[x, x2], outputs=[diff])

kl_loss =  K.sum( - K.square(x - x2) , axis=-1)

vae.add_loss(kl_loss)
vae.compile(optimizer='rmsprop')

vae.fit([x_train, x_train_2],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, x_test_2], None),
        #validation_split=(0.2),
        callbacks=[])

score = vae.evaluate([x_test, x_test_2], verbose=0)
print(score)
'''




'''
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

x_train, y_train_ = reverse_remove_missclass((x_train, y_train_, missing_class))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)
x_train_nonflat = x_train
x_test_nonflat = x_test
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

ye = load_model('full/y_encoder.h5')
mean = load_model('full/mean_encoder.h5')
mean_miss = load_model('miss_0/mean_encoder.h5')

ym = ye.predict(np.array([1,0,0,0,0,0,0,0,0,0]).reshape(1,10))
mean = mean.predict([x_test_nonflat, x_test], batch_size=batch_size)
mean = np.mean(mean, 0)
mean_miss = mean_miss.predict([x_test_nonflat, x_test], batch_size=batch_size)
mean_miss = np.mean(mean_miss, 0)

print(np.sum(ym - mean))
print(np.sum(ym - mean_miss))
'''

'''
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)
x_train_nonflat = x_train
x_test_nonflat = x_test
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


var = load_model('full/var_encoder.h5')
var = var.predict([x_test_nonflat, x_test], batch_size=batch_size)
print(np.sum(var))

var_ori = load_model('ori/var_encoder_ori.h5')
var_ori = var_ori.predict(x_test, batch_size=batch_size)
print(np.sum(var_ori))
'''

'''
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)
x_train_nonflat = x_train
x_test_nonflat = x_test
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




ye = load_model('full/y_encoder.h5')
ye = ye.predict(np.eye(num_classes))
print(ye)

ye_ori = load_model('ori/y_encoder_ori.h5')
ye_ori = ye_ori.predict(np.eye(num_classes))
print(ye_ori)
'''
