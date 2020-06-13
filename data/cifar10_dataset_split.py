#! -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from keras.datasets import cifar10

unseen_class = [0]


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

np.save('cifar10/seen_x.npy', seen_x)
np.save('cifar10/seen_y.npy', seen_y)
np.save('cifar10/unseen_x.npy', unseen_x)
np.save('cifar10/unseen_y.npy', unseen_y)


