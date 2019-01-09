#!/usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def hardlin(data):
    data[data >= -1] = 1
    data[data <  0] = -1
    return data
    

class Nure():
    train_num = 20
    
    def __init__(self, X):
        temp1 = X.dot(X.T)
        temp2 = eye(X.shape[0]) - 1
        temp2[temp2 < 0] = 1
        self.W = 0.1 * temp1 * temp2
        print("权重")
        print(self.W)
        return 
   
    def think(self, X):
        #print(self.W)
        Y = X
        for i in range(self.train_num):
            Y = hardlin(self.W.dot(Y) + X - 1)
        return Y

           
       

Y1 = array([[1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1]]).T
Y2 = array([[-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1]]).T
Y3 = array([[-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1]]).T
Y4 = array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1]]).T
temp = Y1
temp = hstack([temp, Y2])
temp = hstack([temp, Y3])
temp = hstack([temp, Y4])
X = temp # 训练样本
print("训练样本")
print(X)

X1 = array([[1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1]]).T
X2 = array([[-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]]).T
X3 = array([[-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1]]).T
X4 = array([[-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1]]).T
temp = X1
temp = hstack([temp, X2])
temp = hstack([temp, X3])
temp = hstack([temp, X4])
test = temp # 测试数据
print("测试数据")
print(test)

NN = Nure(X)
O = NN.think(X)
print("测试结果")
print(O)

print("测试误差")
print(O - X)

