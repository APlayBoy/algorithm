"""
boosting(提升)算法的一种
"""
import numpy as np


class AdaBoost:
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon  # 分类误差率阈值
        self.alpha = []  # 基本分类器前面的权重/系数
        self.base_list = []  # 基本分类器

    def fit(self, x, y, base_classfier, classfier_params=None):
        """
        根据基分类器和强分类器来构建最终的强分类器
        :param x:
        :param y:
        :param base_classfier:
        :return:
        """
        # 构建最终的强分类器, 暂设输入维度为1
        w = np.full(shape=(y.shape[0],), fill_value=1/y.shape[0])  # 样本的权值,每加入一个基本分类器都要重新计算
        while 1 - np.sum(self.predict(x) == y) / y.shape[0] > self.epsilon:# 分类错误数目占比小于等于epsilon, 停止训练
            base = base_classfier()#定义一个新的基分类器
            base.fit(x, y, **classfier_params)#训练基分类器
            base_predict = base.predict(x)#用基分类器预测
            error_rate = np.sum(np.where(base_predict == y, 0, 1) * w)#计算基分类器带权重的错误率
            if error_rate > 0.5: #基分类器效果不能提升的时候, 停止训练
                break
            alpha = 1.0 / 2 * np.log((1 - error_rate) / error_rate) #根据错误率计算alpha
            self.alpha.append(alpha)
            self.base_list.append(base)
            w *= np.exp(-y * base_predict)#更新w
            w /= np.sum(w)#把w归一化

    def predict(self, x):
        """
        使用组合分类器，计算预测值
        :param x:
        :return:
        """
        score = np.zeros((x.shape[0],))
        for alpha, base in zip(self.alpha, self.base_list):
            score += alpha * base.predict(x)
        return np.sign(score)
