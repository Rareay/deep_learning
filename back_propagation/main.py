#! /usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt 

def logsig(data):
    return (1 / (1 + np.exp(-data)))

def tansig(data):
    return (2 / (1 + np.exp(-(2 * data))) - 1)

def purelin(data):
    return data

def minmax(data):
    data_min = data.min(axis=1)
    data_min = data_min.reshape(data_min.shape[0],1)
    data_max = data.max(axis=1)
    data_max = data_max.reshape(data_max.shape[0],1)
    return np.hstack([data_min, data_max])

def traingdm():
    return 

class Nure():
    __transfer_func = tansig
    __train_func = traingdm

    def __init__(self, X_info, neural_num, transfer_func, train_func):
        X_num = X_info.shape[0]
        self.__func = func
        return

    def test(self, data):
        return self.__func(data)

P = linspace(-10, 10, 200)
X = np.array([[4,2,3],[5,4,6],[6,3,9]])
NN = Nure()
#plt.figure()
#plt.plot(P, NN.test(P))
#plt.show()
print(X)
print(minmax(X))
