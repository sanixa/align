
import numpy as np
np.random.seed(9)

import pandas as pd
import numpy as np
import cv2
from PIL import Image

import os
import re, math
from collections import Counter

from sklearn.neighbors import KDTree

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, load_model

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

def Aligning(plant_LCE):
    #use model to convert LCE to CE
    learned_enc = load_model('/home/uscc/cvamc/model/plant/learned_encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})
    attr_dec = load_model('/home/uscc/cvamc/model/plant/attr_decoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling})

    for i in range(len(plant_LCE)):
        for j in range(len(plant_LCE[0])):
            if plant_LCE[i][j] < 1e-5:
                plant_LCE[i][j] = 0.0
    latent = learned_enc.predict(plant_LCE)

    for i in range(len(latent)):
        for j in range(len(latent[0])):
            if latent[i][j] < 1e-5:
                latent[i][j] = 0.0        
    plant_CE = attr_dec.predict(latent)

    return plant_CE

def Learned_Latent_Class_Embedding(path_list):
    #load image and convert to (128,128,3)
    data = np.empty((len(path_list),IMAGE_SIZE,IMAGE_SIZE,3),dtype="float32")
    for i in range(len(path_list)):
        img = Image.open(path_list[i])
        arr = np.asarray(img,dtype="float32")
        if arr.shape[1] > arr.shape[0]:
            arr = cv2.copyMakeBorder(arr,int((arr.shape[1]-arr.shape[0])/2),int((arr.shape[1]-arr.shape[0])/2),0,0,cv2.BORDER_CONSTANT,value=0)
        else:
            arr = cv2.copyMakeBorder(arr,0,0,int((arr.shape[0]-arr.shape[1])/2),int((arr.shape[0]-arr.shape[1])/2),cv2.BORDER_CONSTANT,value=0)       #长宽不一致时，用padding使长宽一致
        arr = cv2.resize(arr,(IMAGE_SIZE,IMAGE_SIZE))
        if len(arr.shape) == 2:
            temp = np.empty((IMAGE_SIZE,IMAGE_SIZE,3))
            temp[:,:,0] = arr
            temp[:,:,1] = arr
            temp[:,:,2] = arr
            arr = temp
        if arr.shape == (IMAGE_SIZE,IMAGE_SIZE,4):
            temp = np.empty((IMAGE_SIZE,IMAGE_SIZE,3),dtype="float16")
            temp[:,:,0] = arr[:,:,0]
            temp[:,:,1] = arr[:,:,1]
            temp[:,:,2] = arr[:,:,2]
            arr = temp
        if arr.shape != (IMAGE_SIZE,IMAGE_SIZE,3):
            print('convert image error, image_path = ', imgDir)
        data[i,:,:,:] = arr
    #use model to convert image to LCE
    data = data / 255.0
    plant_enc = load_model('/home/uscc/cvamc/model/plant/encoder.h5', custom_objects={'Scaler': Scaler, 'Sampling': Sampling, 'Parm_layer': Parm_layer})
    plant_LCE = plant_enc.predict(data)

    return plant_LCE

def main():

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # SET HYPERPARAMETERS

    global NUM_CLASS, NUM_ATTR, DATASET_PATH, IMAGE_PATH, IMAGE_SIZE
    NUM_CLASS = 38
    NUM_ATTR = 35
    DATASET_PATH = '/home/uscc/New Plant Diseases Dataset(Augmented)/'
    IMAGE_PATH = '/home/uscc/New Plant Diseases Dataset(Augmented)/train/Tomato___healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555_new30degFlipLR.JPG'
    IMAGE_SIZE = 128
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CONVERT PHASE
    
    LCE = Learned_Latent_Class_Embedding([IMAGE_PATH])
    CE = Aligning(LCE)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # EVALUATION OF ZERO-SHOT LEARNING PERFORMANCE

    #load class name and make map dict
    classname = pd.read_csv(DATASET_PATH+'classes.txt',header=None,sep = '\t')
    dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
    dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
    #load arry = class x attr
    class_attr = np.load(DATASET_PATH+'class_attr.npy')
    #query test image true label ( train/['label']/ )
    regex = re.compile(r'train/([A-Za-z()_]*)/')
    match = regex.search(IMAGE_PATH)
    true_label = match.group(1)

    ##query five closest class
    
    tree = KDTree(class_attr)
    dist_5, index_5 = tree.query(CE, k=5)
    pred_labels = [dic_class2name[index] for index in index_5[0]]
    print(pred_labels)
    print(true_label)
    
    ##query five closest image
    SAMPLE_SIZE = 50
    cand_list = []
    for i in range(classname.shape[0]):
        imgDir = DATASET_PATH + 'train/' + classname.loc[i][1] 
        imgs = os.listdir(imgDir)
        indices = list(range(500))
        np.random.shuffle(indices)
        for i in range(SAMPLE_SIZE):
            cand_list.append(imgDir + '/' +imgs[indices[i]])
    image_name_list = cand_list
    cand_list = Learned_Latent_Class_Embedding(cand_list)
    cand_list = Aligning(cand_list)
    
    tree = KDTree(cand_list)
    dist_5, index_5 = tree.query(CE, k=5)
    pred_images = [dic_class2name[int(math.floor(index/SAMPLE_SIZE))] for index in index_5[0]]
    print(pred_images)
    image_name = [image_name_list[index] for index in index_5[0]]
    print(image_name)

if __name__ == '__main__':
    main()
