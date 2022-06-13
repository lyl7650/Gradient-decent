# coding : utf-8
# Logistic Regression

import math,os
import numpy as np
import matplotlib.pyplot as plt


def Sigmoid(theta, X):
    return 1/(1+np.exp(-np.dot(X , theta.T)))

def cost_func(m, X, Y, theta):
    """损失函数"""
    sigmoid = Sigmoid(theta, X)
    case1 = np.multiply(Y.T, np.log(sigmoid))
    case2 = np.multiply((1-Y).T, np.log(1-sigmoid))
    J = -(1/m) * np.sum(case1+case2)
    return J

def gradient_desc(m, X, Y, lr, theta, iter):
    """梯度下降优化损失值对theta优化"""
    # 对J求偏导
    loss = np.zeros(iter)
    for i in range(iter):
        theta = theta - (lr/m) * ((Sigmoid(theta, X).T-Y)*X)
        loss[i] = cost_func(m, X, Y, theta)
        #print("iter:{}, theta:{}, loss_theta:{}".format(i+1, theta[0], round(loss[i], 6)))
    return theta[0], loss

def predict(theta, X, threshold):
    """预测"""
    predict = Sigmoid(theta, X)
    predict = [1 if i > threshold else 0 for i in predict]
    return predict

def plot_loss(loss):
    """绘制损失函数"""
    # 横坐标为迭代次数
    x = np.arange(1, len(loss)+1)
    # 纵坐标为损失函数值
    y = loss
    plt.ylabel("loss")
    plt.xlabel("iter")
    plt.title("loss curve")
    plt.plot(x, y)
    plt.show()


# 计算准确率
def accuracy(Y, predict_y):
    count = 0
    Y = Y.tolist()[0]
    for i in range(len(Y)):
        if int(Y[i]) == predict_y[i]:
            count+=1
    return count/len(Y)

if __name__ == "__main__":
    data = np.loadtxt('./data.txt')
    X = data[:,:2]; Y = np.mat(data[:,2])
    bias = np.ones(data.shape[0])
    X = np.insert(X, 0, values=bias, axis=1)
    X = np.mat(X)
    theta = np.zeros(X.shape[1])
    m = Y.size
    iter = 1000
    threshold = 0.5
    lr = 0.1
    theta, loss = gradient_desc(m, X, Y, lr, theta, iter)
    plot_loss(loss)
    predict_y = predict(theta, X, threshold)
    acc = accuracy(Y, predict_y)
    print("准确率为{}".format(acc))
