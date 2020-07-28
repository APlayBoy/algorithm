import numpy as np
from ml.tree import CART
"""
 1.二分类预测值初始化
    zi = fm(xi)
    pi = 1 / (1 + e**(-zi))
    Loss function:
    L = Sum(-yi * Logp - (1-yi) * Log(1-p))
    Get derivative of p:
    dL / dp = Sum(-yi/p +(1-yi)/(1-p))
    dp / dz = p * (1 - p)
    dL / dz = (dL / dp) * (dp / dz)
    dL / dz = Sum(-yi * (1-p) + (1-yi)* p)
    dL / dz = Sum(p) - Sum(yi) 
    使z的导数等于零，损失函数最小
    p = Mean(yi)
    又1 / (1 + e**(-z)) = Mean(yi)
    所以z = Log(mean(yi) / (1-mean(yi))
2.回归预测值初始化
    loss_function:平方损失含函数
    L = Sum(z - y)**2
    dL / dz = Sum(z - y) = Sum(z) - Sum(y)
    z = Mean(y)
    0-1损失函数：
    指数损失函数
    对数损失函数
"""


class GBDT:

    def __init__(self, learning_rate=0.1, max_tree=5, classfication=True):
        self.learning_rate = learning_rate
        self.classfication = classfication
        self.max_tree = max_tree
        self.trees = []

    def fit(self, x, y, tree_params={}):
        #初始化value
        y = np.where(y == -1, 0, 1) if self.classfication else y
        self.init_value = (lambda _y: np.log(_y / (1 - _y)))(np.mean(y)) if self.classfication else np.mean(y)
        residual = y - (1 / (1 + np.exp(-self.init_value))) if self.classfication else y - self.init_value #计算残差
        while len(self.trees) < self.max_tree:
            tree = CART(classfication=False, **tree_params)
            tree.fit(x, residual, **tree_params)
            self.trees.append(tree)
            value = self.predict_tree_value(x)
            residual = y - (1 / (1 + np.exp(-value))) if self.classfication else y - value

    def predict_tree_value(self, x):
        result = self.init_value
        for tree in self.trees:
            result += self.learning_rate * tree.predict(x)
        return result

    def predict(self, x):
        result = self.init_value
        for tree in self.trees:
            result += self.learning_rate * tree.predict(x)
        if self.classfication:
            result = 1 / (1 + np.exp(-result))
            result = np.where(result > 0.5, 1, -1)
        return result
