#! /usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# �����֪��
class NeuralNetwork():
    train_num = 8000    # ѵ������
    learn_speed = 0.1   # ѧϰ����

    def __init__(self, X_info, neural_num):
        self.X_num = X_info.shape[0]    # �����X�ĸ���
        self.neural_num = neural_num    # ��Ԫ�ĸ���
        self.W = 2 * random.random((self.X_num, self.neural_num)) - 1 # �������Ȩ��

    def __hardlim(self, data):
        data[data >= 0] = 1
        data[data < 0] = -1
        return data

    def think(self, input):
        return self.__hardlim(input.T.dot(self.W)).T

    def train(self, X, D):
        for interation in range(self.train_num):
            O = self.think(X) 
            err = D - O
            self.W += (self.learn_speed * err.dot(X.T)).T
        return 


P = np.array([[1 ,1, 1, 1],[-0.6, -0.7, 0.7, 0.8], [0.1, 1, 0.1, 1]])
T = np.array([[1, 1, 1, -1]])
NN = NeuralNetwork(np.array([[-1, 1],[-1, 1], [-1, 1]]), 1)
NN.train(P, T)
print(NN.think(P))
k = - (NN.W[1] / NN.W[2])
b = - (NN.W[0] / NN.W[2])
xdata = np.linspace(-1, 1)
plt.plot(xdata, xdata*k+b, 'r')
plt.scatter(P[1], P[2], color = 'red', marker = '+')
plt.show()
