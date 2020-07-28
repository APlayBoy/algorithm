"""
实现SoftMax回归，逻辑回归的多分类推广。所以，本质还是一种分类算法, 支持L2正则
"""
import numpy as np


class SoftMaxRegression:
    def __init__(self, l2=1e-4, alpha=0.4):
        self.l2 = l2  # 权值衰减项系数lambda, 类似于惩罚系数
        self.alpha = alpha  # 学习率
        self.w = None  # 权值

    def fit(self, x, y, max_step=10000):
        """
        类的标签，从零开始无间隔的整数，e.g: 0,1,2,3,4...
        :param x:
        :param y:
        :param max_step:
        :return:
        """

        fit_data = np.insert(x, x.shape[1], 1, axis=1) #增加bias的维度
        self.w = np.random.normal(0, 1, (np.unique(y).shape[0], fit_data.shape[1]))  # 针对每个类，都有一组权值参数w
        step = 0
        while step < max_step:
            step += 1
            exp_z = np.exp(fit_data @ self.w.T)
            a = exp_z / exp_z.sum(axis=1).reshape((-1, 1)) #使用softmax激活
            a[np.arange(fit_data.shape[0]), y] -= 1  #计算z的梯度  a(i) - y(i)
            grad = a.T @ fit_data / fit_data.shape[0] + self.l2 * self.w  # 梯度， 第二项为衰减项
            self.w -= self.alpha * grad #使用梯度下降法更新w

    def predict(self, x):
        predict_data = np.insert(x, x.shape[1], 1, axis=1) #增加bias的维度
        return np.argmax(predict_data @ self.w.T, axis=1)

    def predict_prob(self, x):
        predict_data = np.insert(x, x.shape[1], 1, axis=1) #增加bias的维度
        exp_z = np.exp(predict_data @ self.w.T)
        return np.max(exp_z.T / exp_z.sum(axis=1), axis=0)
