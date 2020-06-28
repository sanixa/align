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
import sys, scipy.io

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


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
batch_size = 128
class_input_shape = -1
learned_input_shape = -1
class_output_dim = -1
learned_output_dim = -1

#---------------
argv = sys.argv
dataset = argv[1]


if dataset == 'SUN':
    class_input_shape = (102, )
    learned_input_shape = (256, )
    class_output_dim = 102
    learned_output_dim = 256
elif dataset == 'CUB':
    class_input_shape = (312, )
    learned_input_shape = (256, )
    class_output_dim = 312
    learned_output_dim = 256
elif dataset == 'AwA2':
    class_input_shape = (85, )
    learned_input_shape = (256, )
    class_output_dim = 85
    learned_output_dim = 256
elif dataset == 'plant':
    class_input_shape = (85, )
    learned_input_shape = (256, )
    class_output_dim = 85
    learned_output_dim = 256




enc = load_model('model/' + dataset + '/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})

data = np.load('data/'+ dataset +'/traindata.npy')
attr = np.load('data/'+ dataset +'/trainattr.npy')

data_train, data_test, attr_train, attr_test = train_test_split(data, attr, test_size=0.20, random_state=42)

mean_train = enc.predict([data_train], batch_size=200)
mean_test = enc.predict([data_test], batch_size=200)

latent_dim = 128
intermediate_dim = 128
epochs = 1000




###################input_1#########################
class_embedding_input = Input(shape=class_input_shape)
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
z_class_mean = scaler(z_mean, mode='positive')
z_class_std = scaler(z_std, mode='negative')

sampling = Sampling()
z_class = sampling([z_class_mean, z_class_std])


###################input_2########################
learned_embedding_input = Input(shape=learned_input_shape)
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
z_learned_mean = scaler(z_mean, mode='positive')
z_learned_std = scaler(z_std, mode='negative')

sampling = Sampling(latent_dim)
z_learned = sampling([z_learned_mean, z_learned_std])

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

class_out = class_decoder(z_class)
learned_out = learned_decoder(z_learned)

# 建立模型
vae = Model(inputs=[class_embedding_input, learned_embedding_input], outputs=[class_out, learned_out])



# xent_loss是重构loss，kl_loss是KL loss
xent_loss = 0.5 * K.mean((class_embedding_input - class_out)**2) + \
            0.5 * K.mean((learned_embedding_input - learned_out)**2)
#K.sum(K.categorical_crossentropy(x_in_flat, x_out_flat), axis=-1)

cross_class_out = class_decoder(z_learned)
cross_learned_out = learned_decoder(z_class)
cross_xent_loss = 0.5 * K.mean((class_embedding_input - cross_class_out)**2) + \
                  0.5 * K.mean((learned_embedding_input - cross_learned_out)**2)

kl_loss = - 0.5 * K.sum(1 + z_class_std - K.square(z_class_mean) - K.exp(z_class_std), axis=-1) + \
          - 0.5 * K.sum(1 + z_learned_std - K.square(z_learned_mean) - K.exp(z_learned_std), axis=-1)

align_loss = K.sum(K.square((z_class - z_learned)**2 + \
                            (K.square(K.exp(z_class_std)) - K.square(K.exp(z_learned_std)))**2), axis=-1)



vae_loss = K.mean(xent_loss + cross_xent_loss + kl_loss + align_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)



vae.fit([attr_train, mean_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([attr_test, mean_test], None),
        #validation_split=(0.2),
        callbacks=[history, learning_rate_reduction, early_stopping])#



history.loss_plot('epoch')

attr_encoder = Model(class_embedding_input, z_class)
attr_encoder.save('model/' + dataset + '/attr_encoder.h5')

learned_encoder = Model(learned_embedding_input, z_learned)
learned_encoder.save('model/' + dataset + '/learned_encoder.h5')

class_decoder.save('model/' + dataset + '/attr_decoder.h5')
learned_decoder.save('model/' + dataset + '/learned_decoder.h5')

