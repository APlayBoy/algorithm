"""
使用numpy实现逻辑回归,支持给样本赋予不同的权重
"""

import numpy as np


def sigmoid(z):
    # Logistic函数, 正类的概率
    return 1.0 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, alpha=0.01, l1=None, l2=None):
        self.w, self.b = None, None
        self.alpha = alpha
        self.l1 = l1
        self.l2 = l2

    def loss(self, x, y):
        """
        定义损失函数，梯度下降过程中，不会直接用到
        :return:
        """
        active = self.predict_prob(x)
        loss = np.mean(y * np.log(sigmoid(active)) + (1 - y) * np.log(1 - sigmoid(active)))
        loss = loss + self.l1 * (np.sum(np.abs(self.w)) + np.abs(self.b)) if self.l1 is not None else loss
        loss = loss + self.l2 * (np.sum(np.square(self.w)) + np.square(self.b)) if self.l2 is not None else loss
        return loss

    def fit(self, x, y, sample_weigh=None, max_step=1000):  # 损失函数采用对数损失函数，其数学形式与似然函数一致
        y = np.where(y == -1, 0, y) if np.sum(np.unique(y)) == 0 else y #如果标签是(-1,1)修正为(0,1)
        # 批量梯度下降法
        self.w = np.random.normal(0.0, 1.0, x.shape[1])  # 初始化各特征的权重 按照高斯分布初始化
        self.b = np.int(0)
        i = 0
        while i <= max_step:
            grad = sigmoid(self.w @ x.T + self.b) - y if sample_weigh is None else \
                sample_weigh * (sigmoid(self.w @ x.T + self.b) - y)
            grad_w = np.mean(grad * x.T, axis=1) #使用交叉熵损失函数计算梯度
            grad_b = np.mean(grad)
            if self.l1 is not None:#添加l1正则损失
                grad_w, grad_b = grad_w + self.l1, grad_b + self.l1
            if self.l2 is not None:  #添加l2正则损失
                grad_w, grad_b = grad_w + 2 * self.l2 * self.w, grad_b + 2 *self.l2 * self.b
            w = self.w - self.alpha * grad_w  #更新权重,用所有样本需要更新的梯度的均值更新
            b = self.b - self.alpha * grad_b
            if np.sum(np.abs(self.w - w) + np.abs(self.b - b)) < 0.00001:
                break
            self.w, self.b = w, b
            i += 1

    def predict_prob(self, x):
        return sigmoid(self.w @ x.T + self.b)

    def predict(self, x):
        return np.where(sigmoid(self.w @ x.T + self.b) > 0.5, 1, -1)


