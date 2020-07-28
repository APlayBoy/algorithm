import numpy as np
# class XGBoost:
# 实现XGBoost回归和分类, 以MSE/交叉熵损失函数为例
import numpy as np
from collections import Counter


class Node:
    def __init__(self, sp=None, left=None, right=None, w=None):
        self.sp = sp  # 非叶节点的切分，特征以及对应的特征下的值组成的元组
        self.left = left
        self.right = right
        self.w = w  # 叶节点权重，也即叶节点输出值

    def isLeaf(self):
        return self.w


class Tree:
    def __init__(self, epsilon=0.1, _gamma=0.5, _lambda=0.1, max_depth=5, min_sample=3):
        self.esilon = epsilon
        self._gamma = _gamma  # 正则化项中T前面的系数
        self._lambda = _lambda  # 正则化项w前面的系数
        self.max_depth = max_depth #树的最大深度
        self.min_sample = min_sample #叶子节点中的最小样本数
        self.root = None

    def score(self, garr, harr):
        return - np.square(np.sum(garr)) / (np.sum(harr) + self._lambda) + self._gamma

    def get_weight(self, garr, harr):
        return - np.sum(garr) / (np.sum(harr) + self._lambda)

    def _get_feature_minimun_score(self, x, feature_ind, garr, harr):
        """
        计算该特征(包括离散型和连续型特征）下不同切分点的gini系数,然后返回gini系数最小的切分点
        :param x: 样本的某列数据, shape=(sample_num,)
        :param y: 样本的标签，shape=(sample_num,)
        :param feature_ind: 列的ind
        :return: x中切分点的value, score, ind
        """
        def _child_score(inds):  # 条件score
            return self.score(garr[inds], harr[inds]) + self.score(garr[~inds], harr[~inds])

        unique_value = np.sort(np.unique(x)) if feature_ind in self.continuous else np.unique(x)
        # 计算某个特征及相应的某个特征值组成的切分节点的基尼系数/MSE
        values_score = [_child_score(x <= value if feature_ind in self.continuous else y == value)
                        for value in unique_value]
        min_score_ind = np.argmin(values_score)
        return unique_value[min_score_ind], values_score[min_score_ind]

    def _best_split(self, x, garr, harr):
        """
        根据当前的数据寻找一个全局最佳切分点(基尼系数的变化量最大，也就是信息增益最大)，并根据最佳切分点切分数据
        :param x:
        :param y:
        :return:
        """
        retuple = lambda split_feature, i: None if split_feature is None else (i, split_feature[0], split_feature[1])
        features_score = [retuple(self._get_feature_minimun_score(x[:, d], d, garr, harr), d) for d in np.arange(x.shape[1])]
        feature_index, feature_value, feature_score = min(features_score, key=lambda x: x[2])
        if self.score(garr, harr) - feature_score < self.epsilon:
            return None, None
        return feature_index, feature_value

    def fit(self, x, garr, harr, continuous=[]):
        """
        训练模型,特征可以多次使用
        :param x:
        :param y:
        :param continuous:指明哪些列的数据连续型的，未指明的列默认为离散型
        :return:
        """
        self.root = Node()
        self.continuous = continuous
        stack = [(self.root, x, garr, harr, 1)]
        while len(stack) > 0:
            node, x_data, garr, harr, deep = stack.pop()
            node.result = self.get_weight(garr, harr)
            feature_ind, feature_value = self._best_split(x_data, garr, harr)
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
                stack.append((node.left, x_data[inds], garr[inds], harr[inds], deep+1))
                stack.append((node.right, x_data[~inds], garr[~inds], harr[~inds], deep+1))

    def predict(self, x):
        result = []
        x_data = np.insert(x, x.shape[1], np.arange(x.shape[0]), axis=1)  # 插入标签记录预测结果
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


class XgBoost:
    def __init__(self, epsilon, max_tree, _gamma, _lambda, max_depth, eta=1.0, classfication=True):
        self.epsilon = epsilon  # 最小信息增益
        self.max_tree = max_tree  # 迭代次数，即基本树的个数
        self._gamma = _gamma
        self._lambda = _lambda
        self.max_depth = max_depth  # 单颗基本树最大深度
        self.eta = eta  # 收缩系数, 默认1.0,即不收缩
        self.classfication = classfication
        self.trees = []

    def fit(self, x, y, continuous=[]):
        step = 0
        while step < self.max_tree:
            tree = Tree(self._gamma, self._lambda, self.max_depth)
            y_pred = self.predict(x)
            garr, harr = y_pred - y, np.ones_like(y) #以及导数和二级倒数
            tree.fit(x, garr, harr, continuous)
            self.trees.append(tree)
            step += 1

    def predict(self, x):
        result = 0
        for tree in self.trees:
            result += self.eta * tree.predict(x)
        if self.classfication:
            result = 1 / (1 + np.exp(-result))
            result = np.where(result > 0.5, 1, -1)
        return result


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    f = XgBoost(50, 0.1, 0, 1.0, 4, eta=0.8, classfication=False)
    f.fit(X_train, y_train, continuous=list(range(X_train.shape[1])))
    y_pred = f.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    plt.scatter(np.arange(y_pred.shape[0]), y_test - y_pred)
    plt.show()

