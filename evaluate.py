#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib, sys, os
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, metrics
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
matplotlib.use('TkAgg')
plt.style.use('ggplot')

import keras
from keras.layers import Input, Dense, Lambda, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

unseen_class = [0]

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

#------------------test  func------------------------------


def kmeans(data, label, cluster_num):
    kmeans_fit = KMeans(n_clusters = cluster_num).fit(data)
    cluster_labels = kmeans_fit.labels_
    label = label.reshape(len(label))
    cluster = [[] for x in range(cluster_num)]
    for i in range(len(cluster_labels)):
        cluster[cluster_labels[i]].append(label[i])

    cluster_to_y_label = []
    cluster_to_y_label_intra_cluster_num = []
    for i in range(cluster_num):
        temp = [0 for x in range(cluster_num)]
        for j in range(len(cluster[i])):
            temp[cluster[i][j]] = temp[cluster[i][j]] + 1
        cluster_to_y_label.append(np.argmax(temp))
        cluster_to_y_label_intra_cluster_num.append(temp[np.argmax(temp)])

    print(cluster_to_y_label)
    print(cluster_to_y_label_intra_cluster_num)
    temp = []
    for i in range(cluster_num):
        temp.append(len(cluster[i]))
    print(temp)


def test_cluster_num(data, ks_range):
    silhouette_avgs = []
    for k in ks_range:
        kmeans_fit = cluster.KMeans(n_clusters = k).fit(data)
        cluster_labels = kmeans_fit.labels_
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        silhouette_avgs.append(silhouette_avg)


    plt.bar(ks_range, silhouette_avgs)
    plt.show()
    print(silhouette_avgs)

####---------------------------util--------------------------------------------------

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

def classifier_load_data_binary(dataset, reload_data=0):

    seen_x, seen_y, unseen_x, unseen_y = -1, -1, -1, -1
    num_classes = -1
    if dataset == 'SUN':
        num_classes = 717
    elif dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'plant':
        num_classes = 38
    elif dataset == 'AwA2':
        num_classes = 50
    elif dataset == 'CUB':
        num_classes = 200
    if reload_data == '0':
        data = np.load('data/'+ dataset +'/classifier/classifier_data.npy')
        label = np.load('data/'+ dataset +'/classifier/classifier_label.npy')
    elif reload_data == '1':
    ####---- seen class
        seen_x = np.load('data/'+ dataset +'/traindata.npy')
        seen_y = np.load('data/'+ dataset +'/trainlabel.npy')

    ####---- unseen class
        unseen_x = np.load('data/'+ dataset +'/testdata.npy')
        unseen_y = np.load('data/'+ dataset +'/testlabel.npy')

    ####---- 
        if dataset == 'plant':
            seen_x = seen_x / 255.
            unseen_x = unseen_x / 255.
        ####---- remap label to 0/1
        for i in range(len(seen_y)):
            seen_y[i] = 0
        for i in range(len(unseen_y)):
            unseen_y[i] = 1
        seen_y = to_categorical(seen_y, 2)
        unseen_y = to_categorical(unseen_y, 2)

        enc = load_model('model/'+ dataset + '/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})
        seen_x = enc.predict([seen_x])
        unseen_x = enc.predict([unseen_x])

        data = np.concatenate((seen_x, unseen_x), axis=0)
        label = np.concatenate((seen_y, unseen_y), axis=0)
        np.save('data/'+ dataset +'/classifier/classifier_data.npy', data)
        np.save('data/'+ dataset +'/classifier/classifier_label.npy', label)

    return data, label

def classifier_load_data(dataset, reload_data=0):

    seen_x, seen_y, unseen_x, unseen_y = -1, -1, -1, -1
    num_classes = -1
    if dataset == 'SUN':
        num_classes = 717
    elif dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'plant':
        num_classes = 38
    elif dataset == 'AwA2':
        num_classes = 50
    elif dataset == 'CUB':
        num_classes = 200

    if reload_data == '0':
        seen_x = np.load('data/'+ dataset +'/classifier/seen_classifier_data.npy')
        seen_y = np.load('data/'+ dataset +'/classifier/seen_classifier_label.npy')
        unseen_x = np.load('data/'+ dataset +'/classifier/unseen_classifier_data.npy')
        unseen_y = np.load('data/'+ dataset +'/classifier/unseen_classifier_label.npy')
    elif reload_data == '1':
    ####---- seen class
        seen_x = np.load('data/'+ dataset +'/traindata.npy')
        seen_y = np.load('data/'+ dataset +'/trainlabel.npy')

    ####---- unseen class
        unseen_x = np.load('data/'+ dataset +'/testdata.npy')
        unseen_y = np.load('data/'+ dataset +'/testlabel.npy')

    ####---- 
        if dataset == 'plant':
            seen_x = seen_x / 255.
            unseen_x = unseen_x / 255.
        ####---- remap label to 0~
        label_list =[]
        for idx in seen_y:
            if idx not in label_list:
                label_list.append(idx)
        for i in range(len(seen_y)):
            for j in range(len(label_list)):
                if seen_y[i] == label_list[j]:
                    seen_y[i] = j
        for idx in seen_y:
            if idx > len(label_list):
                print('error')
        seen_y = to_categorical(seen_y, len(label_list))
        ####---- remap label to 0~
        label_list =[]
        for idx in unseen_y:
            if idx not in label_list:
                label_list.append(idx)
        for i in range(len(unseen_y)):
            for j in range(len(label_list)):
                if unseen_y[i] == label_list[j]:
                    unseen_y[i] = j
        for idx in unseen_y:
            if idx > len(label_list):
                print('error')
        unseen_y = to_categorical(unseen_y, len(label_list))
        enc = load_model('model/'+ dataset + '/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})
        

        seen_x = enc.predict([seen_x])
        unseen_x = enc.predict([unseen_x])
    

        np.save('data/'+ dataset +'/classifier/seen_classifier_data.npy', seen_x)
        np.save('data/'+ dataset +'/classifier/seen_classifier_label.npy', seen_y)
        np.save('data/'+ dataset +'/classifier/unseen_classifier_data.npy', unseen_x)
        np.save('data/'+ dataset +'/classifier/unseen_classifier_label.npy', unseen_y)

    return seen_x, seen_y, unseen_x, unseen_y
    
def classifier(data, input_shape, model_path):

    x_train, y_train = data

    #scaler = preprocessing.MinMaxScaler()
    #x_train = scaler.fit_transform(x_train)
    '''
    label_list =[]
    for idx in y_train:
        if np.argmax(idx) not in label_list:
            label_list.append(np.argmax(idx))
    num_classes = len(label_list)
    '''
    x_in = Input(shape=input_shape)
    x = Dense(512, activation='relu')(x_in)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x_in)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x_in)
    x = Dropout(0.2)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=[x_in], outputs=[output])



    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(1e-3),
              metrics=['accuracy'])

    history = LossHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

    model.fit([x_train], [y_train],
        shuffle=True,
        epochs=30,
        batch_size=500,
        #validation_data=([x_test], [y_test]),
        validation_split=(0.2),
        callbacks=[history, learning_rate_reduction])#,early_stopping

    model.save(model_path)
    history.loss_plot('epoch')



def make_plant_attr_file():
    ####----------------given class attr--------------------------

    with open('data/plant/'+'attr.txt', 'r') as f:
        attr = f.readlines()

    for i in range(35):
        attr[i] = attr[i][:-1].split(' ')
        attr[i] = [float(x) for x in attr[i]]
    np.save('data/plant/class_attr.npy',np.array(attr).T)
    attr = np.array(attr).T.tolist()

    ####----------------learned class attr----------compute avg E(x) as learned class attr----------------
    temp_sum_list = [[] for x in range(38)]
    count_list = [0 for x in range(38)]

    seen_data = np.load('/home/uscc/cvamc/data/plant/traindata.npy')
    seen_label = np.load('/home/uscc/cvamc/data/plant/trainlabel.npy')
    unseen_data = np.load('/home/uscc/cvamc/data/plant/testdata.npy')
    unseen_label = np.load('/home/uscc/cvamc/data/plant/testlabel.npy')
    seen_data = seen_data / 255.
    unseen_data = unseen_data / 255.

    enc = load_model('/home/uscc/cvamc/model/plant/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})
            
    seen_data = enc.predict(seen_data)
    unseen_data = enc.predict(unseen_data)

    for i in range(len(seen_data)):
        count_list[seen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[seen_label[i]] == []:
            temp_sum_list[seen_label[i]] = seen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[seen_label[i]] += seen_data[i]

    for i in range(len(unseen_data)):
        count_list[unseen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[unseen_label[i]] == []:
            temp_sum_list[unseen_label[i]] = unseen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[unseen_label[i]] += unseen_data[i]

    for i in range(38):
        temp_sum_list[i] = temp_sum_list[i] / count_list[i]

    temp_sum_list = np.array(temp_sum_list)
    learned_enc = load_model('model/'+ 'plant' + '/learned_encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})
    attr_dec = load_model('model/'+ 'plant' + '/attr_decoder.h5')
    temp_sum_list = learned_enc.predict(temp_sum_list)
    temp_sum_list = attr_dec.predict(temp_sum_list)

    print(temp_sum_list.shape)
    print(temp_sum_list[0])

    np.save('data/'+ 'plant' +'/learned_class_attr.npy', temp_sum_list)


def make_AwA2_attr_file():
    ####----------------given class attr--------------------------
    attr = pd.read_csv('data/AwA2/diff/predicate-matrix-continuous.txt',header=None,sep = '\t')
    attr = [list(filter(('').__ne__, attr.loc[i][0].split(' '))) for i in range(attr.shape[0])]

    for i in range(len(attr)):
        for j in range(len(attr[0])):
            attr[i][j] = float(attr[i][j])

    np.save('data/'+ 'AwA2' +'/class_attr.npy', np.array(attr))

    ####----------------learned class attr----------compute avg E(x) as learned class attr----------------
    temp_sum_list = [[] for x in range(50)]
    count_list = [0 for x in range(50)]

    seen_data = np.load('/home/uscc/cvamc/data/AwA2/traindata.npy')
    seen_label = np.load('/home/uscc/cvamc/data/AwA2/trainlabel.npy')
    unseen_data = np.load('/home/uscc/cvamc/data/AwA2/testdata.npy')
    unseen_label = np.load('/home/uscc/cvamc/data/AwA2/testlabel.npy')

    enc = load_model('/home/uscc/cvamc/model/AwA2/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})

    seen_data = enc.predict(seen_data)
    unseen_data = enc.predict(unseen_data)

    for i in range(len(seen_data)):
        count_list[seen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[seen_label[i]] == []:
            temp_sum_list[seen_label[i]] = seen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[seen_label[i]] += seen_data[i]

    for i in range(len(unseen_data)):
        count_list[unseen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[unseen_label[i]] == []:
            temp_sum_list[unseen_label[i]] = unseen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[unseen_label[i]] += unseen_data[i]

    for i in range(50):
        temp_sum_list[i] = temp_sum_list[i] / count_list[i]

    temp_sum_list = np.array(temp_sum_list)
    learned_enc = load_model('model/'+ 'AwA2' + '/learned_encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})
    attr_dec = load_model('model/'+ 'AwA2' + '/attr_decoder.h5')
    temp_sum_list = learned_enc.predict(temp_sum_list)
    temp_sum_list = attr_dec.predict(temp_sum_list)

    print(temp_sum_list.shape)
    print(temp_sum_list[0])

    np.save('data/'+ 'AwA2' +'/learned_class_attr.npy', temp_sum_list)

def make_SUN_attr_file():
    ### output = num_classes * num_attr

    dataset='SUN'

    ####----------------given class attr--------------------------
    matcontent = scipy.io.loadmat('data/SUN/att_splits.mat')
    attr = matcontent['att'].T
    #for i in range(len(attr)):
      #  for j in range(len(attr[0])):
       #     attr[i,j] = str(attr[i,j])

    np.save('data/'+ dataset +'/class_attr.npy', np.array(attr))

    ####----------------learned class attr----------compute avg E(x) as learned class attr----------------
    temp_sum_list = [[] for x in range(717)]
    count_list = [0 for x in range(717)]

    seen_data = np.load('/home/uscc/cvamc/data/SUN/traindata.npy')
    seen_label = np.load('/home/uscc/cvamc/data/SUN/trainlabel.npy')
    unseen_data = np.load('/home/uscc/cvamc/data/SUN/testdata.npy')
    unseen_label = np.load('/home/uscc/cvamc/data/SUN/testlabel.npy')

    enc = load_model('/home/uscc/cvamc/model/SUN/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})

    seen_data = enc.predict(seen_data)
    unseen_data = enc.predict(unseen_data)

    for i in range(len(seen_data)):
        count_list[seen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[seen_label[i]] == []:
            temp_sum_list[seen_label[i]] = seen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[seen_label[i]] += seen_data[i]

    for i in range(len(unseen_data)):
        count_list[unseen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[unseen_label[i]] == []:
            temp_sum_list[unseen_label[i]] = unseen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[unseen_label[i]] += unseen_data[i]

    for i in range(717):
        temp_sum_list[i] = temp_sum_list[i] / count_list[i]

    temp_sum_list = np.array(temp_sum_list)
    learned_enc = load_model('model/'+ 'SUN' + '/learned_encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})
    attr_dec = load_model('model/'+ 'SUN' + '/attr_decoder.h5')
    temp_sum_list = learned_enc.predict(temp_sum_list)
    temp_sum_list = attr_dec.predict(temp_sum_list)

    print(temp_sum_list.shape)
    print(temp_sum_list[0])

    np.save('data/'+ 'SUN' +'/learned_class_attr.npy', temp_sum_list)

def make_CUB_attr_file():
    ### output = num_classes * num_attr
    ####----------------given class attr--------------------------
    dataset='CUB'
    with open('data/'+ dataset +'/diff/class_attribute_labels_continuous.txt', 'r') as f:
        attr = f.readlines()
    with open('data/'+ dataset +'/diff/image_class_labels.txt', 'r') as f:
        label = f.readlines()
    with open('data/'+ dataset +'/diff/images.txt', 'r') as f:
        images = f.readlines()

    attr_list = [ [] for x in range(200)]

    for i in range(len(images)):#
        img_label = int(label[i].split()[1]) -1 ####label 1~200 ==> 0~199

        if attr_list[img_label] == []:
            temp = attr[img_label].split(' ')
            temp[-1] = temp[-1][:-1]
            temp = [float(x) for x in temp]
            attr_list[img_label] = temp

    np.save('data/'+ dataset +'/class_attr.npy', np.array(attr_list))

    ####----------------learned class attr----------compute avg E(x) as learned class attr----------------
    temp_sum_list = [[] for x in range(200)]
    count_list = [0 for x in range(200)]

    seen_data = np.load('/home/uscc/cvamc/data/CUB/traindata.npy')
    seen_label = np.load('/home/uscc/cvamc/data/CUB/trainlabel.npy')
    unseen_data = np.load('/home/uscc/cvamc/data/CUB/testdata.npy')
    unseen_label = np.load('/home/uscc/cvamc/data/CUB/testlabel.npy')

    enc = load_model('/home/uscc/cvamc/model/CUB/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})

    seen_data = enc.predict(seen_data)
    unseen_data = enc.predict(unseen_data)

    for i in range(len(seen_data)):
        count_list[seen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[seen_label[i]] == []:
            temp_sum_list[seen_label[i]] = seen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[seen_label[i]] += seen_data[i]

    for i in range(len(unseen_data)):
        count_list[unseen_label[i]] +=1
        #### first time, assign data to tem
        if temp_sum_list[unseen_label[i]] == []:
            temp_sum_list[unseen_label[i]] = unseen_data[i]
        ###  sum data to temp
        else:
            temp_sum_list[unseen_label[i]] += unseen_data[i]

    for i in range(200):
        temp_sum_list[i] = temp_sum_list[i] / count_list[i]

    temp_sum_list = np.array(temp_sum_list)
    learned_enc = load_model('model/'+ 'CUB' + '/learned_encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})
    attr_dec = load_model('model/'+ 'CUB' + '/attr_decoder.h5')
    temp_sum_list = learned_enc.predict(temp_sum_list)
    temp_sum_list = attr_dec.predict(temp_sum_list)

    print(temp_sum_list.shape)
    print(temp_sum_list[0])

    np.save('data/'+ 'CUB' +'/learned_class_attr.npy', temp_sum_list)


def diff(dataset='CUB'):

    if dataset == 'CUB':
        num_classes = 200
    elif dataset == 'SUN':
        num_classes = 717
    elif dataset == 'AwA2':
        num_classes = 50
    elif dataset == 'plant':
        num_classes = 38


    attr = np.load('data/'+ dataset +'/class_attr.npy')
    learned_attr = np.load('data/'+ dataset +'/learned_class_attr.npy')
    seen_y = np.load('data/'+ dataset +'/trainlabel.npy')
    unseen_y = np.load('data/'+ dataset +'/testlabel.npy')

    scaler = preprocessing.MinMaxScaler()
    attr = scaler.fit_transform(attr)
    learned_attr = scaler.fit_transform(learned_attr)

    sum_seen_diff = 0
    sum_unseen_diff = 0
    cnt_seen = 0
    cnt_unseen = 0
    for i in range(num_classes):
        attr[i] = attr[i] + np.repeat([1e-1], len(attr[i]))
        if i in seen_y:
            sum_seen_diff += np.sum(np.abs(learned_attr[i] - attr[i]) / attr[i]) / len(attr[i])
            cnt_seen += 1
        elif i in unseen_y:
            sum_unseen_diff += np.sum(np.abs(learned_attr[i] - attr[i]) / attr[i]) / len(attr[i])
            cnt_unseen += 1
        else:
            print('error')

    print('seen class acc : ' + str(sum_seen_diff / cnt_seen))
    print('unseen class acc : ' + str(sum_unseen_diff / cnt_unseen))
        

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    
    #---------------
    argv = sys.argv
    dataset = argv[1]
    method = argv[2]
    reload_data = argv[3]

    method = method.split(',')

    input_shape = -1
    num_classes = -1

    if dataset == 'SUN':
        num_classes = 717
        input_shape = (64,)
    elif dataset == 'cifar10':
        num_classes = 10
        input_shape = (28,)
    elif dataset == 'plant':
        num_classes = 38
        input_shape = (64,)
    elif dataset == 'AwA2':
        num_classes = 50
        input_shape = (64,)
    elif dataset == 'CUB':
        num_classes = 200
        input_shape = (64,)

    '''
    print('================kmeans=====================================')
    data = np.concatenate((mean_seen, mean_unseen))
    label = np.concatenate((y_test, unseen_y))
    print('-----------------test cluster num-------------------')
    ks = range(2, 15)
    test_cluster_num(data, ks)
    print('-----------------kmeans-------------------')
    
    kmeans(data, label, 10)
    '''
    if str(0) in method:
        print('================classifier=====================================')
        data, label = classifier_load_data_binary(dataset, reload_data)
        #seen_x, seen_y, unseen_x, unseen_y = classifier_load_data(dataset, reload_data)
        '''
        print('================seen class acc=====================================')
        classifier([seen_x, seen_y], input_shape, 'model/' + dataset + '/classifier_seen.h5')
        print('================unseen class acc=====================================')
        classifier([unseen_x, unseen_y], input_shape, 'model/' + dataset + '/classifier_unseen.h5')
        '''
        print('================acc=====================================')
        classifier([data, label], input_shape, 'model/' + dataset + '/classifier.h5')
    if str(1) in method:
        print('================difference between CE and LCE=====================================')
        diff(dataset)



main()
#make_AwA2_attr_file()
#make_plant_attr_file()
