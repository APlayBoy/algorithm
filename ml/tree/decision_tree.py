"""
ID3&C4.5决策树算法
"""
from collections import Counter

import numpy as np


class Node:
    """
    构建树的节点类
    """

    def __init__(self, feature=None, value=None, result=None, child=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.child = child  # 特征的每个值对应一颗子树，特征值为键，相应子树为值

    def __repr__(self):
        return "feature: " + str(self.feature)+";value: "+str(self.value)+";result: "+str(self.result)+";"

    def __str__(self):
        return "feature: " + str(self.feature)+";value: "+str(self.value)+";result: "+str(self.result)+";"


class DecisionTree:

    def __init__(self, epsilon=1e-3, min_sample=3, max_depth=5, metric='C4.5'):
        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶节点含有的最少样本数
        self.max_depth = max_depth  # 树的最大深度
        self.tree = None
        self.metric = metric
        self.root = None
        self.min_sample = min_sample  # 叶节点含有的最少样本数
        self.score_func = self._info_gain if self.metric == 'ID3' else self._info_gain_ratio

    @staticmethod
    def _info(y):
        """
        计算经验熵，并返回
        :param y:
        :return:
        """
        label_count = np.array([n for l, n in Counter(y).most_common()])# 统计各个类别的个数
        p = label_count / y.shape[0] #统计各个类别的比例
        return np.sum(-p * np.log2(p)) #计算经验熵

    def _con_info(self, x, y):
        """
        计算条件熵
        :param x:
        :param y:
        :return:
        """
        feature_count = Counter(x).most_common()
        return np.sum([n/y.shape[0] * self._info(y[x == k]) for k, n in feature_count])

    def _info_gain(self, x, y):
        """
        计算信息增益 Gain(x, y) = Info(y) - CondInfo(x, y)
        :param x:
        :param y:
        :return:
        """
        return self._info(y) - self._con_info(x, y)

    def _info_gain_ratio(self, x, y):
        """
        计算信息增益率
        :param feature:
        :param x:
        :param y:
        :return:
        """
        return self._info_gain(x, y) / self._info(x)

    def fit(self, x, y):
        """
        构建树
        :param x:
        :param y:
        :return:
        """
        self.root = Node()
        stack = [(self.root, x, y, 1)]
        while len(stack) > 0:
            node, x_data, y_data, deep = stack.pop()#使用栈模型，先进后出，也就是深度优先构建树
            node.result = Counter(y_data).most_common(1)[0][0] #数据集中出现最多的类作为这个节点的分类结果
            if deep >= self.max_depth or np.unique(y_data).shape[0] == 1 or x_data.shape[0] < self.min_sample:  # 数据集只有一个类或者样本数量少
                continue
            # 获取最优切分特征、相应的信息增益（比）以及切分后的子数据集
            feature_scores = [self.score_func(x_data[:, ind], y_data) for ind in np.arange(x_data.shape[1])]
            max_score_ind = np.argmax(feature_scores)
            if feature_scores[max_score_ind] < self.epsilon:  # 信息增益(比)小于阈值
                continue
            node.feature = max_score_ind
            feature_labels_inds = [(x_data[:, max_score_ind] == v, v) for v in np.unique(x_data[:, max_score_ind])]
            new_x_data = np.delete(x_data, max_score_ind, axis=1)
            new_data = [(Node(value=v), new_x_data[inds], y_data[inds], deep+1) for inds, v in feature_labels_inds]
            node.child = {str(d[0].value): d[0] for d in new_data}
            stack.extend(new_data)

    def predict(self, x):
        if len(x) == 0:
            return []
        x_data = np.insert(x, x.shape[1], np.arange(x.shape[0]), axis=1) #插入标签记录预测结果
        stack = [(self.root, x_data)]
        result = []
        while len(stack) > 0:
            node, node_data = stack.pop()
            if node.child is None:
                result.append(np.array([node_data[:, -1], np.full(node_data.shape[0], node.result)]).T)
                continue
            child_inds = {str(v): np.argwhere(node_data[:, node.feature] == v).reshape(-1)
                          for v in np.unique(node_data[:, node.feature])}
            node_data = np.delete(node_data, node.feature, axis=1)
            result.extend([np.array([node_data[ind, -1], np.full(node_data[ind, -1].shape[0], node.result)]).T
                           for v, ind in child_inds.items() if node.child.get(v) is None])#处理样本中出现的新value
            stack.extend([(node.child.get(v), node_data[ind, :]) for v, ind in child_inds.items()
                          if node.child.get(v) is not None])
        result = np.vstack(result)
        return result[result[:, 0].argsort()][:, -1].astype(np.int)

