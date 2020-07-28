"""
朴素贝叶斯分类算法使用numpy实现
采用后验期望计参数，先验概率分布取均匀分布
目前暂不支持离散型特征的数据
"""
import numpy as np


class NBayes:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_  # 贝叶斯估计参数lambda
        self.p_prior = None  # 模型的先验概率, 注意这里的先验概率不是指预先人为设定的先验概率，而是需要估计的P(y=Ck)
        self.p_condition = None  # 模型的条件概率
        self.dim_index = None #数据离散后所在的新维度
        self.is_binary_classification = True#二分类(标签为-1，1)还是多分类器(标签为0,1,2,3..)

    def fit(self, x, y):
        """
        默认所有的数据都是离散数据，并且类别标签从零开始无间断排列
        二分类时候标签为-1和1
        支持多分类但是所有类别的标签需要从0开始无间断排列，e.g: 0，1，2，3这样
        :param x:
        :param y:
        :return:
        """
        y_num = y.shape[0]
        self.is_binary_classification = np.unique(y).shape[0] == 2
        y = np.where(y == -1, 0, y) if self.is_binary_classification else y #而分类器的标签
        # 统计先验概率
        self.p_prior = np.bincount(y)
        self.p_prior = (self.p_prior + self.lambda_) / (y_num + self.p_prior.shape[0] * self.lambda_)
        #统计条件概率
        dims = np.zeros(x.shape[1], dtype=np.int)
        for d in range(x.shape[1]):
            dims[d] = np.unique(x[:, d]).shape[0]
        self.p_condition = np.zeros((self.p_prior.shape[0], dims.sum()), dtype=np.float)#记录条件概率
        lambda_vec = np.zeros(dims.sum(), dtype=np.float)#记录条件概率每个维度分母的lambda的值
        self.dim_index = np.insert(np.cumsum(dims), 0, 0)[:-1]#记录x中每个维度在onehot后新维度的起始位置
        for i, n in zip(self.dim_index, dims):
            lambda_vec[i:i+n] = n * self.lambda_
        for l in range(self.p_prior.shape[0]):
            l_x = x[np.where(y == l)] + self.dim_index
            self.p_condition[l] = (np.bincount(l_x.reshape((-1))) + self.lambda_) / (lambda_vec + l_x.shape[0])

    def predict(self, x):
        """
        :param x: np.ndarray
        :return:
        """
        predict_data = x if len(x.shape) == 2 else np.expand_dims(x, 0)
        score = np.argmax(np.prod(self.p_condition[:, predict_data], axis=2).T * self.p_prior, axis=1)
        return np.where(score == 0, -1, score) if self.is_binary_classification else score

    def predict_prob(self, x):
        predict_data = x if len(x.shape) == 2 else np.expand_dims(x, 0)
        prob = np.prod(self.p_condition[:, predict_data], axis=2).T * self.p_prior
        return prob / prob.sum()
