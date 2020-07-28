import numpy as np
from scipy.stats import multivariate_normal

"""
实现了高斯混合模型的训练和预测
"""


class GMM(object):
    """
    每个样本的对数似然函数为 Pi(1) * pdf(x, mu(1), var(1)) +Pi(2) * pdf(x, mu(2), var(2)) ....
    """

    def __init__(self, n_cluster=None, dim=None, mus=None, vars=None):
        self.n_cluster = n_cluster
        self.dim = dim
        self.mus = mus
        self.vars = vars

    def fit(self, x, n_cluster=3, params=None, step=20):
        self.n_cluster = n_cluster
        fit_data = x if isinstance(x, np.ndarray) else np.array(x)
        self.dim = fit_data.shape[1]
        if params is None:#初始化参数
            min_x = np.min(fit_data, axis=0)
            max_x = np.max(fit_data, axis=0)
            self.mus = np.random.random([self.n_cluster, self.dim]) * (max_x - min_x) + min_x #随机初始mus
            self.vars = np.ones_like(self.mus, dtype=np.float) #方差设置为零
            self.pi = np.ones(self.n_cluster) / self.n_cluster
        else:
            self.mus, self.vars, self.pi = params["mus"], params["vars"], params["pi"]

        for s in range(step):
            gamma = np.zeros((fit_data.shape[0], self.n_cluster), dtype=np.float)
            #E step
            for c in range(n_cluster):
                gamma[:, c] = multivariate_normal.pdf(fit_data, self.mus[c], np.diag(self.vars[c]))
            gamma = gamma * self.pi
            gamma = (gamma.T / np.sum(gamma, axis=1)).T

            expect = np.sum(gamma, axis=0) #属于各个高斯分布的期望
            #M step
            #更新Pi
            self.pi = expect / fit_data.shape[0]
            #更新mu
            for i in range(self.n_cluster):
                self.mus[i] = np.average(fit_data, axis=0, weights=gamma[:, i])
            #更新vars
            for i in range(self.n_cluster):
                self.vars[i] = np.average((fit_data - self.mus[i]) ** 2, axis=0, weights=gamma[:, i])

    def predict(self, x):
        predict_data = np.array(x)
        if len(predict_data.shape) == 1:
            predict_data = np.expand_dims(predict_data, axis=0)
        gamma = np.zeros((predict_data.shape[0], self.n_cluster), dtype=np.float)
        for c in range(self.n_cluster):
            gamma[:, c] = multivariate_normal.pdf(predict_data, self.mus[c], np.diag(self.vars[c]))
        gamma = gamma * self.pi
        result = np.argmax(gamma, axis=1)
        return result[0] if result.shape[0] == 1 else result
