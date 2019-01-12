#!/usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random
import numpy.linalg as lg


# 欧式范数
# ||x|| = (|x1|^2 + |x2|^2 + |x3|^2 + ... + |xn|^2)^0.5
def radbas(data):
    X = data
    for i in range(X.shape[0]):
        X[i] = X[i] ** 2
    for i in range(X.shape[0]):
        if i == 0:
            Y = X[i]
        else:
            Y += X[i]
    Y = Y ** 0.5
    return Y

def classification(data, centers):
    X = data * 1
    C = centers * 1
    X = X.dot(ones((1,C.shape[1])))
    D = radbas(X - C)
    D_min = D.min()
    for i in range(C.shape[1]):
        if D[i] == D_min:
            D = zeros((C.shape[1], 1))
            D[i][0] = 1
            break
    return D
        

def creat_center(data, num):
    samples_num = data.shape[1] # 样本数目
    if samples_num < num:
        return None
    X = data * 1
    #print("全部样本：")
    #print(X)

    # 把前面几个样本作为样本中心，这只是样本中心的初始值，后面还会更改 
    temp = random.sample(range(X.shape[1]), num)
    for i in range(num):
        if i == 0:
            C = array([X[:,temp[0]]]).T
        else:
            C = hstack([C, array([X[:,temp[i]]]).T])
    C = C * 1.0
    #print("初始样本中心：")
    #print(C)

    while 1:
        # 根据欧式距离将输入样本分类，分类信息存放于矩阵U
        for i in range(samples_num):
            u = classification(array([X[:,i]]).T, C)
            if i == 0:
                U = u
            else:
                U = hstack([U, u])
        #print("分类矩阵：")
        #print(U)
                
        for i in range(num):
            u = U[i] 
            c_temp = zeros((X.shape[0], 1))
            u_num = 0
            for j in range(samples_num):
                if u[j] == 1:
                    u_num += 1
                    c_temp += array([X[:,j]]).T
            c_temp = c_temp / u_num
            if i == 0:
                C_TEMP = c_temp
            else:
                C_TEMP = hstack([C_TEMP, c_temp])
                
        if (C == C_TEMP).all():
            break
        else:
            C = C_TEMP * 1.0
    #print("最终样本中心：")
    #print(C)

    #plt.scatter(X[0], X[1], color='r')
    #plt.scatter(C[0], C[1], color='b')
    #plt.show()

    return C


class Nure():

    def __init__(self, samples, d_out, cen_num):
        X = samples * 1.0
        D = d_out * 1.0
        N_I = X.shape[0]
        N_H = cen_num
        N_O = D.shape[0]
        print("样本：")
        print(X)
        print("期望：")
        print(D)
        C =  creat_center(X, N_H)
        self.C = C * 1.0
        print("样本中心：")
        print(C)

        self.__sigma = self.get_sigma(C)
        #self.__sigma = 0.50
        print("sigma = ", self.__sigma)

        O_H = self.hidden_output(X, C)
        print("O_H:")
        print(O_H)

        D = D.T
        print("D:")
        print(D)
        
        O_H_ = lg.pinv(O_H)
        print("O_H_:")
        print(O_H_)

        W = O_H_.dot(D)
        self.W = W * 1.0
        print("W = O_H_ * D")
        print(W)

        O_O = O_H.dot(W)
        print("D = O_H * W")
        print(O_O)
        
        
    def get_sigma(self, sample_certen):
        C = sample_certen * 0.1
        C_num = C.shape[1]
        C_max = 0
        for i in range(C_num):
            for j in range(i + 1, C_num):
                temp = array([C[:,i] - C[:,j]]).T
                temp = radbas(temp)
                if C_max < temp:
                    C_max = temp
        return C_max / ((2 * C_num) ** 0.5)


    def hidden_output(self, sample, sample_certen):
        X = sample * 1.0
        C = sample_certen * 0.1
        for i in range(X.shape[1]):
            X_i = array([X[:,i]]).T * 1.0
            X_i = X_i.dot(ones((1,C.shape[1])) * 1.0)
            D = radbas(X_i - self.C)
            temp = exp(-1 / (2 * self.__sigma * self.__sigma) * D)
            if i == 0:
                O = temp
            else:
                O = vstack([O, temp])
        return O


    def think(self, samples):
        X = samples * 1.0
        O_H = self.hidden_output(X, self.C)
        O_O = O_H.dot(self.W)
        return O_O

 



#P = array([[0,0.2],[1,4.2],[0,1.1],[7,8],[9.1,8],[8.3,9],[9.4,2],[8.2,2],[9,1.2]]).T
#T = np.random.rand(10, 1)
#T = array([[2,2],[3,3],[4,4],[2,2],[3,3],[4,4],[2,2],[3,3],[4,4]]).T
P = array([linspace(100,200,10)])
T = 100 * np.random.rand(10, 1).T
#creat_center(a, 3)
NN = Nure(P, T, 10) # 训练样本、期望输出、隐神经元个数

test = array([linspace(100,200,100)])
out = NN.think(test)
plt.plot(test[0], out.T[0])
plt.scatter(P, T, color = 'r')
plt.show()

