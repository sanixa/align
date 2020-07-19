#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import norm
import os, sys

import keras
from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout
from keras.layers import Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

class Scaler(keras.layers.Layer):
    """特殊的scale层
    """
    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * K.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * K.sigmoid(-self.scale)
        return inputs * K.sqrt(scale)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Sampling(keras.layers.Layer):
    def __init__(self, latent_dim=128, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_config(self):
        base_config = super(Sampling, self).get_config()
        config = {'latent_dim': self.latent_dim}
        return dict(list(base_config.items()) + list(config.items()))

class Parm_layer(keras.layers.Layer):
    def __init__(self, ratio=0.5, **kwargs):
        super(Parm_layer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        super(Parm_layer, self).build(input_shape)

    def call(self, inputs):
        m1, m2 = inputs

        return self.ratio * m1 + (1 - self.ratio) *m2
        
    def get_config(self):
        base_config = super(Parm_layer, self).get_config()
        config = {'ratio': self.ratio}
        return dict(list(base_config.items()) + list(config.items()))

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
batch_size = 64
input_shape = -1
kernel_size = 3
filters = 64
latent_dim = 64
intermediate_dim = 128
epochs = 1000
num_classes = -1

argv = sys.argv
dataset = argv[1]
if dataset == 'SUN':
    num_classes = 717
    input_shape = (128, 128, 3)
elif dataset == 'cifar10':
    num_classes = 10
    input_shape = (28, 28, 3)
elif dataset == 'plant':
    num_classes = 38
    input_shape = (128, 128, 3)
elif dataset == 'AwA2':
    num_classes = 50
    input_shape = (128, 128, 3)
elif dataset == 'CUB':
    num_classes = 200
    input_shape = (128, 128, 3)


seen_x = np.load('data/'+ dataset +'/traindata.npy')
seen_y = np.load('data/'+ dataset +'/trainlabel.npy')


if dataset == 'plant':
    seen_x = seen_x / 255.


x_train, x_test, y_train, y_test = train_test_split(seen_x, seen_y, test_size=0.20, random_state=412)

np.save('data/'+ dataset +'/x_train.npy', x_train)
np.save('data/'+ dataset +'/x_test.npy', x_test)
np.save('data/'+ dataset +'/y_train.npy', y_train)
np.save('data/'+ dataset +'/y_test.npy', y_test)


#x_train = np.load('data/'+ dataset +'/x_train.npy')
#x_test = np.load('data/'+ dataset +'/x_test.npy')
#y_train = np.load('data/'+ dataset +'/y_train.npy')
#y_test = np.load('data/'+ dataset +'/y_test.npy')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)



x_in = Input(shape=input_shape)
x_in_shape = K.int_shape(x_in)
x = x_in
###############input_1#########################
mc1 = load_model('resnet50.h5')
mc1_x = mc1(x)
mc1_h = Dense(intermediate_dim, activation='relu')(mc1_x)
mc1_h = Dropout(0.5)(mc1_h)
###############input_2#########################
for i in range(3):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               padding='same')(x)
    x = LeakyReLU(0.2)(x)

# 备份当前shape，等下构建decoder的时候要用
shape = K.int_shape(x)
x = Flatten()(x)
h = Dense(intermediate_dim, activation='relu')(x)
h = Dropout(0.5)(h)
###################input_3#########################
y_in = Input(shape=(num_classes,)) # 输入类别
y = Dense(intermediate_dim)(y_in)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(intermediate_dim)(y_in)
y = Dropout(0.5)(y)
yh = Dense(latent_dim)(y) # 这里就是直接构建每个类别的均值




# 算p(Z|X)的均值和方差
mc1_z_mean = Dense(latent_dim)(mc1_h)
mc1_z_log_var = Dense(latent_dim)(mc1_h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

mc1_z_mean = BatchNormalization()(mc1_z_mean)
mc1_z_log_var = BatchNormalization()(mc1_z_log_var)
z_mean = BatchNormalization()(z_mean)
z_log_var = BatchNormalization()(z_log_var)

scaler = Scaler()
mc1_z_mean = scaler(mc1_z_mean, mode='positive')
mc1_z_log_var = scaler(mc1_z_log_var, mode='negative')
z_mean = scaler(z_mean, mode='positive')
z_log_var = scaler(z_log_var, mode='negative')


save_ratio = 0.2 ## save ratio for origin image
parm_layer = Parm_layer(save_ratio)
z_plus_mean = parm_layer([z_mean, mc1_z_mean])
z_plus_log_var = parm_layer([z_log_var, mc1_z_log_var])

sampling = Sampling(latent_dim)
z = sampling([z_plus_mean, z_plus_log_var])

# 解码层，也就是生成器部分
# 先搭建为一个独立的模型，然后再调用模型
latent_inputs = Input(shape=(latent_dim,))
#x = Dense(intermediate_dim, activation='relu')(latent_inputs)
#x = BatchNormalization()(x)
#x = Dropout(0.2)(x)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(3):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
    x = LeakyReLU(0.2)(x)
    filters //= 2

x = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='relu',
                          padding='same')(x)

x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(x_in_shape[1] * x_in_shape[2] * x_in_shape[3], activation='relu')(x)

# 搭建为一个独立的模型
decoder = Model(latent_inputs, outputs)

x_out = decoder(z)

# 建立模型
vae = Model(inputs=[x_in, y_in], outputs=[x_out, yh])

for layer in vae.layers:
    if layer.name == 'resnet50':
        layer.trainable = False


x_in_flat = Flatten()(x_in)
#x_out_flat = Flatten()(x_out)
# xent_loss是重构loss，kl_loss是KL loss

xent_loss = 0.5 * K.sum(K.categorical_crossentropy(x_in_flat, x_out), axis=-1)#K.mean((x_in_flat - x_out_flat)**2)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_plus_log_var - K.square(z_plus_mean - yh) - K.exp(z_plus_log_var), axis=-1)

#cos相似度的loss,保證類別向量散度
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.RMSprop(1e-2))
vae.summary()

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)



vae.fit([x_train, y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None),
        #validation_split=(0.2),
        callbacks=[history, learning_rate_reduction, early_stopping])



history.loss_plot('epoch')

encoder = Model(x_in, z)
encoder.save('model/' + dataset +'/encoder.h5')

decoder.save('model/' + dataset + '/decoder.h5')

mu = Model(y_in, yh)
mu.save('model/' + dataset + '/y_encoder.h5')

