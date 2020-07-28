from .base import Base
import numpy as np
"""
实现了三类问题：1.概率计算 2.学习问题（参数估计） 3.预测问题（状态序列的预测）
把乘法转换成log运算
"""


class LogHMM(Base):

    def __init__(self, n_q, n_v, Pi=None, A=None, B=None):
        Base.__init__(self, n_q, n_v, Pi=Pi, A=A, B=B)

    def gen_one_data(self, step):
        """
        根据隐马尔科夫模型的参数随机生成相应的观测数据
        """
        #初始化第一个隐藏态
        Pi, A, B = np.exp(self.Pi), np.exp(self.A), np.exp(self.B)
        state = [np.min(np.argwhere(np.cumsum(Pi) > np.random.random()))]
        obs = [np.min(np.argwhere(np.cumsum(B[state[-1]]) > np.random.random()))]
        #生成第一个观测序列
        for _ in range(1, step):
            state.append(np.min(np.argwhere(np.cumsum(A[state[-1]]) > np.random.random())))
            obs.append(np.min(np.argwhere(np.cumsum(B[state[-1]]) > np.random.random())))
        return state, obs

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
        _t = len(obs) - 1 if t is None else t
        for i in range(_t + 1):
            alpha = (self.Pi if i == 0 else np.log(np.sum(np.exp(alpha + self.A.T), axis=1))) + self.B[:, obs[i]]
        return alpha

    def forward_prob(self, obs):
        """
        使用前向算法计算obs的概率
        :param obs:
        :return:
        """
        _obs = obs if isinstance(obs, np.ndarray) else np.array(obs)
        return np.sum(np.exp(self._alpha(obs)))

    def _beta(self, obs, t=None):
        """
        计算时刻t,各个状态的后向概率， 【后向是条件概率】
        beta(t-1)(i) = bete(t)(1) * a(i, 1) * b(1, o(t)) + bete(t)(2) * a(i, 2) * b(2, o(t))
        + bete(t)(3) * a(i, 3) * b(3, o(t)) + ... + bete(t)(n_1) * a(i, n_q) * b(n_q, o(t))
        :param obs:
        :param t:
        :return:
        """
        t = 0 if t is None else t
        for i in range(len(obs)-t):  # 注意求beta(t)(i) 使用的是 obs(t+1)
            beta = np.zeros(shape=self.n_q, dtype=np.float) if i == 0 else\
                np.log(np.sum(np.exp(self.A + self.B[:, obs[len(obs) - i]] + beta), axis=1))
        return beta

    def backward_prob(self, obs):
        """
        使用后向算法计算obs的概率
        :param obs:
        :return:
        """
        _obs = obs if isinstance(obs, np.ndarray) else np.array(obs)
        return np.sum(np.exp(self.Pi + self._beta(obs) + self.B[:, obs[0]]))

    def _gamma(self, obs, t):
        """
        计算时刻t处于各状态的概率
        :param obs:
        :return:
        """
        prob = np.exp(self._alpha(obs, t) + self._beta(obs, t))
        return prob / np.sum(prob)

    def fb_prob(self, obs, t=None):

        # 使用后向算法和后向算法计算obs的概率
        t = int(len(obs) / 2) if t is None else t
        return np.sum(np.exp(self._alpha(obs, t) + self._beta(obs, t)))

    def _xi(self, obs, t): #ksi 求转移概率
        """
        计算再第t步，转移的概率 ksi(t)(i, j) = alpha(t)(i) * A(i,j) * B(j, o(t+1) ) * beta(t+1)(j)
        :param obs:
        :param t:
        :return:`
         """
        alpha_t = self._alpha(obs, t)
        beta_t = self._beta(obs, t + 1)
        obs_beta = self.B[:, obs[t + 1]] + beta_t
        alpha_obs_beta = np.tile(alpha_t, (self.n_q, 1)).T + np.tile(obs_beta, (self.n_q, 1))
        xi = np.exp(alpha_obs_beta + self.A)
        return xi / xi.sum()

    def fit(self, obs_data, max_step=1000, init=True):
        # 利用Baum-Welch算法学习
        #初始化需要学习的参数
        self.A = np.log(np.ones((self.n_q, self.n_q)) / self.n_q) if init or self.A is None else self.A
        self.B = np.log(np.ones((self.n_q, self.n_v)) / self.n_v) if init or self.B is None else self.B
        if init or self.Pi is None:
            self.Pi = np.random.sample(self.n_q)  # 初始状态概率矩阵（向量），的初始化必须随机状态，否则容易陷入局部最优
            self.Pi = np.log(self.Pi / self.Pi.sum())
        step = 0
        while step < max_step: #循环次数
            xi = np.zeros_like(self.A) #记录所有样本中ksi的期望
            gamma = np.zeros_like(self.Pi) #记录所有样本中gamma的期望
            gamma_b = np.zeros_like(self.B) #记录所有样本中gamma_b的期望
            Pi = np.zeros_like(self.Pi)
            gamma_end = np.zeros_like(self.Pi) #记录每个样本最后一个步的gamma值
            for obs in obs_data: #一次循环每个样本
                Pi += self._gamma(obs, 0)
                for t in range(len(obs) - 1):
                    tmp_gamma = self._gamma(obs, t)
                    gamma += tmp_gamma #gamma的期望
                    xi += self._xi(obs, t) #ksi的期望
                    gamma_b[:, obs[t]] += tmp_gamma #的期望
                #最后一步没有ksi,只计算gama_end即可
                tmp_gamma_end = self._gamma(obs, len(obs) - 1)
                gamma_b[:, obs[-1]] += tmp_gamma_end
                gamma_end += tmp_gamma_end
            # 更新 A
            self.A = np.log(xi / gamma.reshape((-1, 1)))
            # 更新 B
            self.B = np.log(gamma_b / (gamma + gamma_end).reshape(-1, 1))
            # 更新 Pi
            self.Pi = np.log(Pi / len(obs_data))
            step += 1

    def _viterbi(self, obs):
        if len(obs) == 0:
            return []
        delta = self.Pi + self.B[:, obs[0]]
        memory = np.zeros((self.n_q, len(obs)), dtype=int)  # 存储时刻t且状态为i时， 前一个时刻t-1的状态，用于构建最终的状态序列

        for i in range(len(obs)-1):
            p = delta + self.A.T
            memory[i] = np.argmax(p, axis=1)
            delta = p[np.arange(self.n_q), memory[i]] * self.B[:, obs[i+1]]
        path = []
        path.append(np.argmax(delta))
        for i in range(len(memory)-2, -1, -1):
            path.append(memory[i, path[-1]])
        path.reverse()
        return path

    def predict(self, obs):
        return self._viterbi(obs)
