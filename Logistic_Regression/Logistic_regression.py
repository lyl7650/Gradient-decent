# coding : utf-8

# Logistic Regression

import math,os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("./Logistic_Regression")


def Sigmoid(theta, X):
    return 1/(1+np.exp(-X*theta.T))

def cost_func(m, X, Y, theta):
    """损失函数"""
    sigmoid = Sigmoid(theta, X)
    J = (-1/m)*np.sum(Y*np.log(sigmoid) + (1-Y)*np.log(1-sigmoid)) # 当前theta下的损失值
    return J

def gradient_desc(m, X, Y, lr, theta, iter):
    """梯度下降优化损失值，对theta优化"""
    # 对J求偏导
    loss = np.zeros(iter)
    for i in range(iter):
        theta = theta - lr*(1/m)*np.sum((1/(1+np.exp(-X*theta.T))-Y).T*X)
        loss[i] = cost_func(m, X, Y, theta)
        print("iter:{}, theta:{}, loss_theta:{}".format(i+1, theta, loss[i]))
    return theta, loss

# 预测
def predict(theta, X, threshold):
    predict = Sigmoid(theta, X)
    predict = [1 if i > threshold else 0 for i in predict]
    return predict

# 计算准确率
def accuracy(Y, predict_y):
    count = 0
    for i in range(Y.size):
        if int(Y[i]) == predict_y[i]:
            count+=1
    return count/Y.size



if __name__ == "__main__":
    data = np.loadtxt('./data.txt', dtype=float)
    X = data[:,:2]
    Y = data[:,2]
    theta = np.mat(np.zeros(X.shape[1]))
    m = Y.size
    iter = 500
    threshold = 0.5
    lr = 0.002
    theta, loss = gradient_desc(m, X, Y, lr, theta, iter)
    predict_y = predict(theta, X, threshold)
    acc = accuracy(Y, predict_y)
    print("准确率为{}".format(acc))
    #print("Y true:{}; Y predict:{}".format(Y.tolist(),predict_y))
