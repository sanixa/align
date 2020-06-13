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
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys

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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
batch_size = 50
class_input_shape = -1
learned_input_shape = -1
class_output_dim = -1
learned_output_dim = -1
#---------------
argv = sys.argv
if argv[1] == '':
    class_input_shape = 
    learned_input_shape = 
    class_output_dim = 
    learned_output_dim = 

latent_dim = 64
intermediate_dim = 64
epochs = 1000


path = 'data/plant/'

###################input_1#########################
class_embedding_input = Input(shape=input_shape)
x = class_embedding_input
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
z_mean = Dense(latent_dim, activation='relu')(x)
z_std = Dense(latent_dim, activation='relu')(x)

scaler = Scaler()
z_mean = scaler(z_mean, mode='positive', name='Scaler')
z_std = scaler(z_std, mode='negative', name='Scaler')

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z_class = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_std])

###################input_2########################
learned_embedding_input = Input(shape=input_shape)
x = learned_embedding_input
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
z_mean = Dense(latent_dim, activation='relu')(x)
z_std = Dense(latent_dim, activation='relu')(x)

scaler = Scaler()
z_mean = scaler(z_mean, mode='positive', name='Scaler')
z_std = scaler(z_std, mode='negative', name='Scaler')

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z_learned = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_std])

# 解码层，也就是生成器部分
# 先搭建为一个独立的模型，然后再调用模型
#--------------------------
class_latent_inputs = Input(shape=(latent_dim,))
x = class_latent_inputs
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
class_output = Dense(class_output_dim, activation='relu')(x)

learned_latent_inputs = Input(shape=(latent_dim,))
x = learned_latent_inputs
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
learned_output = Dense(learned_output_dim, activation='relu')(x)
#------------------------------------------------------------
# 搭建为一个独立的模型
class_decoder = Model(class_latent_inputs, class_output)
learned_decoder = Model(learned_latent_inputs, learned_output)

class_out = class_decoder(z)
learned_out = learned_decoder(z)

# 建立模型
vae = Model(inputs=[class_embedding_input, learned_embedding_input], outputs=[class_out, learned_out])



x_in_flat = Flatten()(x_in)
x_out_flat = Flatten()(x_out)
# xent_loss是重构loss，kl_loss是KL loss
xent_loss = 0.5 * K.sum(K.binary_crossentropy(x_in_flat, x_out_flat), axis=-1)#K.mean((x_in_flat - x_out_flat)**2)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_plus_log_var - K.square(z_plus_mean - yh) - K.exp(z_plus_log_var), axis=-1)

#cos相似度的loss,保證類別向量散度
cos_loss = cos_similarity#* 2e-5
vae_loss = K.mean(xent_loss + kl_loss + cos_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)



vae.fit([x_train, y_train, y_class_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test, y_class_test], None),
        #validation_split=(0.2),
        callbacks=[history, learning_rate_reduction, early_stopping])



history.loss_plot('epoch')

mean_encoder = Model(x_in, z_plus_mean)
mean_encoder.save('model/plant/mean_encoder.h5')

var_encoder = Model(x_in, z_plus_log_var)
var_encoder.save('model/plant/var_encoder.h5')

decoder.save('model/plant/generator.h5')

mu = Model(y, yh)
mu.save('model/plant/y_encoder.h5')
