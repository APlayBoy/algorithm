"""
CART分类回归树，是一颗二叉树，以某个特征以及该特征对应的一个值为节点，故相对ID3算法，最大的不同就是特征可以使用多次
"""
from collections import Counter
import numpy as np


class Node:
    def __init__(self, feature=-1, value=None, result=None, right=None, left=None):
        self.feature = feature  # 特征
        self.value = value  # 特征对应的值
        self.result = result  # 叶节点标记
        self.right = right
        self.left = left


class CART:
    def __init__(self, epsilon=1e-3, min_sample=3, max_depth=5, classfication=True, **param):
        self.epsilon = epsilon
        self.continuous = [] #记录哪些特征是连续型的
        self.min_sample = min_sample  # 叶节点含有的最少样本数
        self.max_depth = max_depth #树的最大深度
        self.root = None
        self.classfication = classfication #分类还是回归
        self.score = self._gini if self.classfication else self._mse
        self.param = param

    @staticmethod
    def _gini(y):
        """
        计算基尼系数
        :param y:
        :return:
        """
        return 1 - np.sum(np.square([np.sum(y == value) / y.shape[0] for value in np.unique(y)]))

    def _mse(self, y):
        return 0 if y.shape[0] == 0 else np.mean(np.square(y - np.mean(y)))

    def _get_feature_minimun_score(self, x, y, feature_ind):
        """
        计算该特征(包括离散型和连续型特征）下不同切分点的gini系数,然后返回gini系数最小的切分点
        :param x: 样本的某列数据, shape=(sample_num,)
        :param y: 样本的标签，shape=(sample_num,)
        :param feature_ind: 列的ind
        :return: x中切分点的value, score, ind
        """
        def con_score_fun(inds): #条件score
            return np.sum(inds)/x.shape[0] * self.score(y[inds]) + np.sum(~inds)/x.shape[0] * self.score(y[~inds])

        unique_value = np.sort(np.unique(x)) if feature_ind in self.continuous else np.unique(x)
        # 计算某个特征及相应的某个特征值组成的切分节点的基尼系数/MSE
        values_score = [con_score_fun(x <= value if feature_ind in self.continuous else y == value)
                        for value in unique_value]
        min_score_ind = np.argmin(values_score)
        return unique_value[min_score_ind], values_score[min_score_ind]

    def _best_split(self, x, y):
        """
        根据当前的数据寻找一个全局最佳切分点(基尼系数的变化量最大，也就是信息增益最大)，并根据最佳切分点切分数据
        :param x:
        :param y:
        :return:
        """
        retuple = lambda split_feature, i: None if split_feature is None else (i, split_feature[0], split_feature[1])
        features_score = [retuple(self._get_feature_minimun_score(x[:, d], y, d), d) for d in np.arange(x.shape[1])]
        feature_index, feature_value, feature_score = min(features_score, key=lambda x: x[2])
        if self.score(y) - feature_score < self.epsilon:
            return None, None
        return feature_index, feature_value

    def fit(self, x, y, continuous=[], **param):
        """
        训练模型,特征可以多次使用
        :param x:
        :param y:
        :param continuous:指明哪些列的数据连续型的，未指明的列默认为离散型
        :return:
        """
        self.param = param
        self.root = Node()
        self.continuous = continuous
        stack = [(self.root, x, y, 1)]
        while len(stack) > 0:
            node, x_data, y_data, deep = stack.pop()
            node.result = Counter(y_data).most_common(1)[0][0] if self.classfication else np.mean(y_data)
            feature_ind, feature_value = self._best_split(x_data, y_data)
            if feature_ind is None or deep >= self.max_depth:
                continue
            else:
                inds = x_data[:, feature_ind] <= feature_value if feature_ind in self.continuous else \
                    x_data[:, feature_ind] == feature_value
                if np.sum(inds) < self.min_sample or np.sum(~inds) < self.min_sample:
                    continue#叶子节点样本数太少，则不分裂
                node.feature = feature_ind
                node.value = feature_value
                node.left = Node()
                node.right = Node()
                stack.append((node.left, x_data[inds], y_data[inds], deep+1))
                stack.append((node.right, x_data[~inds], y_data[~inds], deep+1))

    def predict(self, x):
        result = []
        x_data = np.insert(x, x.shape[1], np.arange(x.shape[0]), axis=1) #插入标签记录预测结果
        stack = [(self.root, x_data)]
        while len(stack) > 0:
            node, node_data = stack.pop()
            if node.feature < 0:
                result.append(np.array([node_data[:, -1], np.full(node_data.shape[0], node.result)]).T)
                continue
            inds = node_data[:, node.feature] <= node.value if node.feature in self.continuous \
                else node_data[:, node.feature] == node.value
            stack.append((node.left, node_data[inds]))
            stack.append((node.right, node_data[~inds]))

        result = np.vstack(result)
        result = result[result[:, 0].argsort()][:, -1]#按照ind排序
        return result.astype(np.int) if self.classfication else result
