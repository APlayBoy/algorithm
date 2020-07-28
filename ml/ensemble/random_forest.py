"""
随机森林算法，组合算法bagging(装袋)的一种
"""
from collections import defaultdict
import numpy as np
import math
from ..tree import CART


class RandomForest:

    def __init__(self, num_tree=6, num_feature=None, ri=True, L=None, classfication=True):
        self.num_tree = num_tree
        self.num_feature = num_feature  # 每棵树中特征的数量
        self.ri = ri  # 判定特征的选择选用RI还是RC, 特征比较少时使用RC(随机挑选L个特征的随机权重组合构成一个新特征)
        self.L = L# 选择ri=False时，进行线性组合的特征个数
        self.classfication=classfication
        self.trees = []  # 随机森林中子树的list

    def extract_features(self, x):
        self.num_feature = int(math.log2(self.D) + 1) if self.num_feature is None else self.num_feature# 默认选择特征的个数
        if self.ri:# 从原数据中抽取特征(RI)或线性组合构建新特征(RC)
            feature_array = np.random.choice(x.shape[1], self.num_feature, replace=False)
        else:
            feature_array = np.zeros((self.num_feature, x.shape[1]))
            weight = np.random.uniform(-1, 1, (self.num_feature, x.shape[1]))
            inds = np.apply_along_axis(lambda row: np.random.choice(row.shape[0], self.L, replace=False), 1, weight)
            feature_array[np.arange(feature_array.shape[0]), inds.T] += 1
            feature_array = feature_array * weight
        return feature_array

    def extract_data(self, x, y):
        # 从原数据中有放回的抽取样本，构成每个决策树的自助样本集
        feature_array = self.extract_features(x)
        inds = np.unique(np.random.choice(x.shape[0], x.shape[0]))  # 有放回抽取样本
        train_x = x[inds]
        train_y = y[inds]
        train_x = train_x[:, feature_array] if self.ri else train_x @ feature_array.T
        return train_x, train_y, feature_array

    def fit(self, x, y, tree_params):
        # 训练主函数
        for i in range(self.num_tree):
            sample_x, sample_y, sample_feature = self.extract_data(x, y)
            subtree = CART(classfication=self.classfication, **tree_params)
            subtree.fit(sample_x, sample_y, **tree_params)
            self.trees.append((subtree, sample_feature))  # 保存训练后的树及其选用的特征，以便后续预测时使用
        return

    def predict(self, x):
        """
        :param x: ndarray
        :return:
        """
        if self.classfication:#预测，多数表决
            result = [defaultdict(int) for _ in range(x.shape[0])]  # 存储每个类得到的票数
            for tree, features in self.trees:
                for di, r in zip(result, tree.predict(x[:, features] if self.ri else x @ features.T)):
                    di[r] += 1
            return np.array([max(r, key=r.get) for r in result])
        else:#预测，求均值
            result = 0
            for tree, features in self.trees:
                result += tree.predict(x[:, features] if self.ri else x @ features.T)
            return result / self.num_tree
