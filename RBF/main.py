#!/usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random
import numpy.linalg as lg
import mpl_toolkits.mplot3d

def radbas(data):
    ''' 欧式范数
    ||x|| = (|x1|^2 + |x2|^2 + |x3|^2 + ... + |xn|^2)^0.5
    '''
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
    ''' 样本分类器
    计算样本到各个样本中心的距离，根据距离最近原则将其分类到
    对应的样本中心

    Args:
        data: 单个输入样本，形状N*1
        centers: 所有样本中心，形状N*M

    Returns:
        D: 分类矩阵[[0] 表示分类到第2个样本中心类中 
                    [1]
                    [0]]
    '''
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
    ''' 生成聚类中心

    Args: 
        data: 所有样本，形状 N*M
        num: 生成的聚类中心个数，num <= M 

    Returns: 
        C: 聚类中心，形状 N*num 
    '''
    samples_num = data.shape[1] # 样本数目
    if samples_num < num:
        return None
    X = data * 1
    #print("全部样本：")
    #print(X)

    # 随机初始化样本中心，即从样本中随机选取num个作为初始样本中心 
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
    return C


class Nure():

    def __init__(self, samples, d_out, cen_num):
        X = samples * 1.0
        D = d_out * 1.0
        print("样本：")
        print(X)

        print("期望：")
        print(D.T)

        C =  creat_center(X, cen_num)
        self.C = C * 1.0
        print("样本中心：")
        print(C)

        self.__sigma = self.get_sigma(C)
        print("sigma = ", self.__sigma)

        O_H = self.hidden_output(X, C)
        print("O_H:")
        print(O_H)

        O_H_ = lg.pinv(O_H)
        print("O_H_:")
        print(O_H_)

        W = O_H_.dot(D)
        self.W = W * 1.0
        print("W = O_H_ * D")
        print(W)

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
        return C_max / ((2 * C_num) ** 0.5)


    def hidden_output(self, sample, sample_certen):
        ''' 计算隐藏层的输出
        '''
        X = sample * 1.0
        C = sample_certen * 1.0
        for i in range(X.shape[1]):
            X_i = array([X[:,i]]).T * 1.0
            X_i = X_i.dot(ones((1,C.shape[1])) * 1.0)
            D = radbas(X_i - C)
            temp = np.exp(-1 / (2 * self.__sigma ** 2) * (D ** 2))
            if i == 0:
                O = temp
            else:
                O = vstack([O, temp])
        return O

    def think(self, samples):
        ''' 计算网络输出，提供给外部使用
        '''
        X = samples * 1.0
        O_H = self.hidden_output(X, self.C)
        O_O = O_H.dot(self.W)
        return O_O

test_num = 1
if test_num == 1:
    ####################### 测试1 ##########################
    P = array([linspace(100,200,10)])
    T = np.sin(P / 10).T
    #T = 100 * np.random.rand(10, 1)
    NN = Nure(P, T, P.shape[1]) # 训练样本、期望输出、隐神经元个数
    test_data = array([linspace(100,200,100)])
    out = NN.think(test_data)
    plt.plot(test_data[0], out.T[0])
    plt.scatter(P, T, color = 'r')
    plt.show()

elif test_num == 2:
    ####################### 测试2 ##########################
    temp = array([linspace(1,100,10)])
    temp = ones((temp.shape[1], 1)).dot(temp)
    temp1 = temp.reshape(temp.shape[1] ** 2)
    temp2 = temp.T.reshape(temp.shape[1] ** 2)
    P = vstack([temp2, temp1])
    T = 10 * np.random.rand(temp.shape[1] ** 2, 1)

    NN = Nure(P, T, temp.shape[1] ** 2)
    
    temp = array([linspace(1,100,100)])
    temp = ones((temp.shape[1], 1)).dot(temp)
    x = temp
    y = temp.T
    temp1 = temp.reshape(temp.shape[1] ** 2)
    temp2 = temp.T.reshape(temp.shape[1] ** 2)
    test_data = vstack([temp2, temp1])
    out = NN.think(test_data)
    z = out.reshape(temp.shape[1], temp.shape[1])
    
    #三维图形
    ax = plt.subplot(111, projection='3d')
    ax.set_title('RBF test');
    ax.plot_surface(x,y,z,rstride=2, cstride=1, cmap=plt.cm.Blues_r)
    #设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    print(P[0])
    print(P[1])
    print(T.T[0])
    ax.scatter(P[0], P[1], T.T[0], color = 'r')
    plt.show()



