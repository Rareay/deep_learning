#!/usr/bin/python3
# coding=gbk

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
 

def creat_C(C, C_size, C_range):
    C_size = C_size.astype(int)
    for i in range(C_size[0][0]):
        temp = C_range[0][0] + (C_range[0][1] - C_range[0][0]) * i / (C_size[0][0] - 1)
        if i == 0:
            C_temp = array([[temp]])
        else:
            C_temp = hstack([C_temp, array([[temp]])])
    if C.shape[1] != 1:
        for i in range(C.shape[1]):
            temp = vstack([array([C[:,i]]).T.dot(ones((1, C_temp.shape[1]))) ,C_temp])
            if i == 0:
                C_next = temp
            else:
                C_next = hstack([C_next, temp])
    else:
        C_next = C_temp
    C_end = C_next
    if C_size.shape[1] > 1:
        C_size_next = C_size.dot(vstack([zeros((1, C_size.shape[1] - 1)), eye(C_size.shape[1] - 1)]))
        C_range_next = hstack([zeros((C_range.shape[0] - 1, 1)) ,eye(C_range.shape[0] - 1)]).dot(C_range)
        C_end = creat_C(C_next, C_size_next, C_range_next)
    return C_end


class Nure():
    ''' 自组织神经网络
    用于分类，类似于RBF实验中的寻找聚类中心
    '''
    train_num = 50
    learn_speed = 0.01
    sigma = 1

    def __init__(self, samples, certen_size):
        X = samples * 1.0
        C_size = array([certen_size])
        self.X_range = vstack([array([X.min(axis = 1)]),array([X.max(axis = 1)])]).T
        C = eye(1)
        self.C = creat_C(C, C_size, self.X_range)
        self.sigma = self.get_sigma(self.C)
        #plt.scatter(samples[0], samples[1], color = 'b')
        #plt.scatter(C[0], C[1], color = 'r')
        #plt.show()

    def get_sigma(self, sample_certen):
        ''' 获取高斯函数的参数sigma
        sigma = C_max / ((2 * C_num) ** 0.5)
        其中，C_max为各个样本中心间最大的距离，
        C_num为样本中心的个数
        '''
        C = sample_certen * 1.0
        C_num = C.shape[1]
        C_max = 0
        for i in range(C_num):
            for j in range(i + 1, C_num):
                temp = array([C[:,i] - C[:,j]]).T
                temp = radbas(temp)
                if C_max < temp:
                    C_max = temp
        return C_max / ((0.25 * C_num) ** 0.5)
        #return C_max / ((2 * C_num) ** 0.5)


    def find_winner(self, sample, centers):
        '''寻找获胜神经元 
        '''
        X = sample * 1.0
        C = centers * 1.0
        X = X.dot(ones((1,C.shape[1])))
        D = radbas(X - C)
        D_min = D.min()
        for i in range(C.shape[1]):
            if D[i] == D_min:
                break
        return i
        #return array([C[:,i]]).T


    def train(self, samples):
        X = samples * 1.0
        C = self.C * 1.0
        for num_train in range(self.train_num):
            for i in range(X.shape[1]):
                learn_speed = self.learn_speed * exp(-0.5 * num_train / self.train_num)
                sigma = self.sigma * exp(-4 * num_train / self.train_num)
                winner_num = self.find_winner(array([X[:,i]]).T, C)
                winner = array([C[:,winner_num]]).T
                for j in range(C.shape[1]):
                    C_temp = array([C[:,j]]).T
                    h = exp(-1 * ((radbas(winner - C_temp)) ** 2) / (sigma ** 2))
                    if h > 0.01:
                        C[:,j] += learn_speed * (C[:,winner_num] - C[:,j])
        plt.scatter(X[0], X[1], color = 'b')
        plt.scatter(C[0], C[1], color = 'r')
        plt.show()




P = np.random.rand(2,4)
P1 = P * 1.0
P1[0] = P1[0] +4
P = np.random.rand(2,4)
P2 = P * 1.0
P2[1] = P2[1] +4
P = np.random.rand(2,4)
P3 = P * 1.0
P3 = P3 +2
P = np.random.rand(2,4)
P4 = P * 1.0
P4 = P4 +4
P = np.random.rand(2,4)
P5 = P * 1.0
P = hstack([P1, P2, P3, P4, P5])

#P = array([[1,2,3],[1,2,3],[1,2,3]])
#P = np.random.rand(3,3)
NN = Nure(P, array([10, 10]))
NN.train_num = 100
NN.train(P)
#NN.train_num = 20
#out = NN.find_certen()
#
#plt.scatter(P[0], P[1], color = 'b')
#plt.scatter(out[0], out[1], color = 'r')
#plt.show()
