#!/usr/bin/python3
# coding=utf-8

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random

def radbas(data):
    ''' 欧式范数
    ||x|| = (|x1|^2 + |x2|^2 + |x3|^2 + ... + |xn|^2)^0.5
    '''
    X = data * 1.0
    for i in range(X.shape[0]):
        X[i] = X[i] ** 2
    for i in range(X.shape[0]):
        if i == 0:
            Y = X[i]
        else:
            Y += X[i]
    Y = Y ** 0.5
    return Y


class Nure():
    ''' 自组织神经网络
    用于分类，类似于RBF实验中的寻找聚类中心
    '''
    train_num = 50
    learn_speed = 0.5

    def __init__(self, samples, certen_num):
        self.X = samples * 1.0
        X_range = vstack([array([self.X.min(axis = 1)]),array([self.X.max(axis = 1)])]).T

        # 随机初始化数据中心
        for i in range(self.X.shape[0]):
            temp = X_range[i][0] + (X_range[i][1] - X_range[i][0]) * np.random.rand(1, certen_num)
            if i == 0:
                C = temp
            else:
                C = vstack([C, temp])
        self.C = C

    def find_certen(self):
        '''寻找聚类中心
        '''
        for i in range(self.train_num):
            for j in range(self.X.shape[1]):
                X_j = array([self.X[:,j]]).T
                X_j = X_j.dot(ones((1, self.C.shape[1])))
                D = radbas(X_j - self.C)
                D_min = D.min()
                for m in range(D.shape[0]):
                    if D_min == D[m]:
                        self.C[:,m] += self.learn_speed * (self.X[:,j] - self.C[:,m])
                        break
            self.learn_speed *= 0.9
        C = self.C * 1.0
        return C

P = np.random.rand(2,10)
P1 = P * 1.0
P1[0] = P1[0] +2
P = np.random.rand(2,10)
P2 = P * 1.0
P2[1] = P2[1] +2
P = np.random.rand(2,10)
P3 = P * 1.0
P3 = P3 +2
P = np.random.rand(2,10)
P4 = P * 1.0
P4 = P4 +1
P = np.random.rand(2,10)
P5 = P * 1.0
P = hstack([P1, P2, P3, P4, P5])

NN = Nure(P, 5)
NN.train_num = 20
out = NN.find_certen()

plt.scatter(P[0], P[1], color = 'b')
plt.scatter(out[0], out[1], color = 'r')
plt.show()
