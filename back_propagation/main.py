#! /usr/bin/python3
# coding=gbk

import numpy as np
from numpy import *
import matplotlib.pyplot as plt 

# 激活函数 logsig
def logsig(data):
    return (1 / (1 + np.exp(-data)))

# 激活函数 tansig
def tansig(data):
    return (2 / (1 + np.exp(-(2 * data))) - 1)

# 激活函数 purelin
def purelin(data):
    return data

# 获取矩阵最大最小值
def minmax(data):
    data_min = data.min(axis=0)
    data_min = data_min.reshape(data_min.shape[0],1)
    data_max = data.max(axis=0)
    data_max = data_max.reshape(data_max.shape[0],1)
    return np.hstack([data_min, data_max])

def traingdm():
    return 


class Nure():
    __transfer_func = None
    __train_func = None
    __layer = None
    learn_speed = 0.5
    train_num = 1000

    def __init__(self, X_info, neural_num, *arg_list):
        self.X_num = X_info.shape[0]
        self.__layer = neural_num.shape[1]
        self.__transfer_func = ones((1, self.__layer), dtype = 'O')
        for i in range(self.__layer):
            self.__transfer_func[0][i] = logsig #激活函数都为tansig
        self.__train_func = traingdm
        arg_num = 2
        for arg in arg_list: 
            if arg is not None:
                if arg_num == 2:
                    self.__transfer_func = arg
                elif arg_num ==3:
                    self.__train_func = arg
        self.W1 = random.rand(self.X_num + 1, neural_num[0][0])
        self.W2 = random.rand(neural_num[0][0] + 1, neural_num[0][1])
        #print("W1")
        #print(self.W1)
        #print("W2")
        #print(self.W2)
        return

    def think(self, X):
        temp = -1 * ones((X.shape[0],1))
        X = hstack([X, temp])
        self.H = self.__transfer_func[0][0](dot(X, self.W1))
        temp = -1 * ones((self.H.shape[0],1))
        self.H = hstack([self.H, temp])
        Y = self.__transfer_func[0][1](dot(self.H, self.W2))
        return Y

    def train_once(self, X, D): # 单次训练
        temp = -1 * ones((X.shape[0],1))
        X = hstack([X, temp])
        H = self.__transfer_func[0][0](dot(X, self.W1))
        temp = -1 * ones((H.shape[0],1))
        H = hstack([H, temp])
        Y = self.__transfer_func[0][1](dot(H, self.W2))
        G = Y * (Y - 1) * (Y - D)
        self.W2 += self.learn_speed * H.T.dot(G)
        E = H * (1 - H) * self.W2.dot(G.T).T
        temp = vstack([eye(E.shape[1] - 1), zeros((1, E.shape[1] - 1))])
        E = dot(E, temp)
        self.W1 += self.learn_speed * X.T.dot(E)
    
    def train(self, X, D): # 训练输入样本
        for i in range(self.train_num):
            for j in range(X.shape[0]):
                self.train_once(array([X[j]]), array([D[j]]))
        return 


#P = array([[1,2,3,4],[3,4,5,6],[5,6,7,8]])
#T = array([[0.2,0.2,0.2],[0.3,0.3,0.3],[0.7,0.7,0.7]])
#P = array([[0,0],[0,1],[1,0],[1,1]])
#T = array([[0],[1],[1],[0]])
P = array([linspace(-10, 10, 10)]).T
T = random.rand(10,1)
#NN = Nure(minmax(P), array([[3,1]]), array([[tansig, purelin]]), traingdm)
NN = Nure(minmax(P), array([[20,1]]))
NN.train_num = 50000;
NN.learn_speed = 0.7
NN.train(P, T)

P_test = array([linspace(-10, 10, 100)]).T
Y_test = NN.think(P_test)
# 绘制测试数据
plt.plot(P_test.T[0], Y_test.T[0])
# 绘制训练样本
plt.scatter(P.T[0], T.T[0], color = 'red', marker = '+')
plt.show()
