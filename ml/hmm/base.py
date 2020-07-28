#!/usr/bin/env python3
import numpy as np
"""
实现了三类问题：1.概率计算 2.学习问题（参数估计） 3.预测问题（状态序列的预测）
直接使用乘法进行计算
"""

class Base:

    def __init__(self, n_q, n_v, Pi=None, A=None, B=None):
        self.n_q = n_q  # 隐藏状态的种类
        self.n_v = n_v  # 观测状态得种类
        self.Pi = Pi if Pi is None else (Pi if isinstance(Pi, np.ndarray) else np.array(Pi))  # 1*n_q, 初始状态概率向量
        self.A = A if A is None else (A if isinstance(A, np.ndarray) else np.array(A))  # n_q*n_q, 状态转移概率矩阵
        self.B = B if B is None else (B if isinstance(B, np.ndarray) else np.array(B))  # n_q*n_v, 观测生成概率矩阵

    def gen_one_data(self, step):
        """
        根据隐马尔科夫模型的参数随机生成相应的观测数据
        """
        pass

    def gen_data(self, num, min_len=5, max_len=10):
        data = []
        for _ in range(num):
            data.append(self.gen_one_data(np.random.randint(min_len, max_len))[1])
        return data

    def _alpha(self, obs, t=None):
        """
        计算时刻t, 各个状态的前向概率，【前向是联合概率】
        alpha(t+1)(i) =
        alpha(t)(1) * a(1, i) * b(i, o(t+1)) + alpha(t)(2) * a(2, i) * b(i, o(t+1)) +
         alpha(t)(3) * a(3, i) * b(i, o(t+1)) + ...+alpha(t)(n_q) * a(n_q, i) * b(i, o(t+1))
        :param
        obs: 观测状态
        :param
        t: 第t步, 默认计算到最后一步, 按照数组下标，第一步t = 0
        :return: 第t步的前向概率
        """
        pass

    def forward_prob(self, obs):
        """
        使用前向算法计算obs的概率
        :param obs:
        :return:
        """
        pass

    def _beta(self, obs, t=None):
        """
        计算时刻t,各个状态的后向概率， 【后向是条件概率】
        beta(t-1)(i) = bete(t)(1) * a(i, 1) * b(1, o(t)) + bete(t)(2) * a(i, 2) * b(2, o(t))
        + bete(t)(3) * a(i, 3) * b(3, o(t)) + ... + bete(t)(n_1) * a(i, n_q) * b(n_q, o(t))
        :param obs:
        :param t:
        :return:
        """
        pass

    def backward_prob(self, obs):
        """
        使用后向算法计算obs的概率
        :param obs:
        :return:
        """
        pass

    def _gamma(self, obs, t):
        """
        计算时刻t处于各状态的概率
        :param obs:
        :return:
        """
        pass

    def fb_prob(self, obs, t=None):
        # 使用后向算法和后向算法计算obs的概率
        pass

    def _xi(self, obs, t): #ksi 求转移概率
        """
        计算再第t步，转移的概率 ksi(t)(i, j) = alpha(t)(i) * A(i,j) * B(j, o(t+1) ) * beta(t+1)(j)
        :param obs:
        :param t:
        :return:`
         """
        pass

    def fit(self, obs_data, maxstep=1000, init=True):
        # 利用Baum-Welch算法学习
        pass

    def _viterbi(self, obs):
        """
        使用viterbi算法去计算隐藏状态
        :param obs:
        :return:
        """
        pass

    def predict(self, obs):
        return self._viterbi(obs)

