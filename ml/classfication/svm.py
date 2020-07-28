"""
支持向量机
本程序求解时采用序列最小最优化算法(SMO)
"""
import numpy as np


class SVM:
    def __init__(self, epsilon=1e-5, C=1.0, kernel=None, kernel_params={}):
        self.epsilon = epsilon
        self.C = C #软间隔的惩罚系数
        self.kernel = kernel  # 是否选择核函数
        self.kernel_params = kernel_params  # 核方法的参数
        self.x = None  # 训练数据集
        self.y = None  # 类标记值，是计算w,b的参数，故存入模型中
        self.alpha = None  # 1*n 存储拉格朗日乘子, 每个样本对应一个拉格朗日乘子
        self.b = 0  # 阈值b, 初始化为0
        self._dot_function = self._dot()

    def _dot(self):
        """
        根据核函数参数生成计算内积的方法
        :return:
        """
        def make_kernel(x, z=None):
            z = x if z is None else z
            if self.kernel == 'linear':
                return np.dot(x, z.T)
            elif self.kernel == 'poly':
                return (np.dot(x, z.T) + 1.0) ** self.kernel_params["degree"]
            elif self.kernel == 'gaussian' or self.kernel == 'rbf':
                return np.exp(-self.kernel_params["gamma"] * (-2.0 * np.dot(x, z.T) + np.sum(x * x, axis=1).reshape((-1, 1)) +
                              np.sum(z * z, axis=1)))
            else:
                return np.dot(x, z.T)
        return make_kernel

    @staticmethod
    def select_second_alpha(ind1, error):
        """
        根据第一个alpha, 挑选第二个变量alpha, 返回索引
        存在预测误差的样本作为候选样本
        不选择ind1
        选择 abs|E1-E2| 最大的样本
        """
        ind2 = np.argmax(np.where(error == 0, 0, 1) * np.abs(error - error[ind1]))
        if not error[ind2] == [ind1]:
            return ind2
        ind2 = np.random.choice(error.shape[0] - 1) # 随机选择一个不与ind1相等的样本索引
        return ind2+1 if ind2 >= ind1 else ind2

    def update(self, ind1, ind2, error, x_dot):
        # 更新挑选出的两个样本的alpha、对应的预测值及误差和阈值b
        old_alpha1, old_alpha2 = self.alpha[ind1], self.alpha[ind2]
        y1, y2 = self.y[ind1], self.y[ind2]

        if y1 == y2:
            L = max(0.0, old_alpha2 + old_alpha1 - self.C)
            H = min(self.C, old_alpha2 + old_alpha1)
        else:
            L = max(0.0, old_alpha2 - old_alpha1)
            H = min(self.C, self.C + old_alpha2 - old_alpha1)
        if L == H:
            return 0
        e1, e2 = error[ind1], error[ind2]
        k11, k12, k22 = x_dot[ind1, ind1], x_dot[ind1, ind2], x_dot[ind2, ind2]
        # 先更新alpha2
        eta = k11 + k22 - 2 * k12
        if eta <= 0:
            return 0
        new_unc_alpha2 = old_alpha2 + y2 * (e1 - e2) / eta  # 未经剪辑的alpha2
        if new_unc_alpha2 > H:
            new_alpha2 = H
        elif new_unc_alpha2 < L:
            new_alpha2 = L
        else:
            new_alpha2 = new_unc_alpha2
        # 再更新alpha1
        if abs(old_alpha2 - new_alpha2) < self.epsilon * (
                old_alpha2 + new_alpha2 + self.epsilon):  # 若alpha2更新变化很小，则忽略本次更新
            return 0
        new_alpha1 = old_alpha1 + y1 * y2 * (old_alpha2 - new_alpha2)
        self.alpha[[ind1, ind2]] = [new_alpha1, new_alpha2]
        # 更新阈值b
        new_b1 = -e1 - y1 * k11 * (new_alpha1 - old_alpha1) - y2 * k12 * (new_alpha2 - old_alpha2) + self.b
        new_b2 = -e2 - y1 * k12 * (new_alpha1 - old_alpha1) - y2 * k22 * (new_alpha2 - old_alpha2) + self.b
        if 0 < new_alpha1 < self.C:
            self.b = new_b1
        elif 0 < new_alpha2 < self.C:
            self.b = new_b2
        else:
            self.b = (new_b1 + new_b2) / 2
        # 更新对应的预测误差
        error[ind1] = np.sum(self.y * self.alpha * x_dot[ind1, :]) + self.b - y1
        error[ind2] = np.sum(self.y * self.alpha * x_dot[ind2, :]) + self.b - y2
        return 1

    def satisfy_kkt(self, y, err, alpha):
        """
        在精度范围内判断是否满足KTT条件
        r = y * err,  r<=0,则y(g-y)<=0,yg<1, alpha=C则符合；r>0,则yg>1, alpha=0则符合
        :param y:
        :param err:预测结果
        :param alpha:
        :return:
        """
        if (y*err < -self.epsilon and alpha < self.C) or (y * err > self.epsilon and alpha > 0):
            return
        return True

    def fit(self, x, y, max_step=500):
        """
        训练主函数,使用smo算法求解
        启发式搜索第一个alpha时，当间隔边界上的支持向量全都满足KKT条件时，就搜索整个数据集。
        整个训练过程需要在边界支持向量与所有样本集之间进行切换搜索，以防止无法收敛
        :param x:
        :param y:
        :param max_step:
        :return:
        """
        # 初始化参数, 包括核内积矩阵、alpha和预测误差

        x_dot = self._dot_function(x)
        self.x, self.y = x, y
        self.alpha = np.zeros(x.shape[0])#alpha全部初始化为0
        error = -y  # 则w,b(所有样本计算的均值)为0， 则相应的预测值初始化为0，预测误差就是-y
        entire_set, step, change_pairs = True, 0, 0
        while step < max_step and (change_pairs > 0 or entire_set):  # 当搜寻全部样本，依然没有改变，则停止迭代
            change_pairs = 0
            if entire_set:  # 搜索整个样本集
                for ind1 in range(self.x.shape[0]):
                    if not self.satisfy_kkt(y[ind1], error[ind1], self.alpha[ind1]):
                        ind2 = self.select_second_alpha(ind1, error)
                        change_pairs += self.update(ind1, ind2, error, x_dot)
            else:  # 搜索间隔边界上的支持向量(bound_search)
                bound_inds = np.where((0 < self.alpha) & (self.alpha < self.C))[0]
                for ind1 in bound_inds:
                    if not self.satisfy_kkt(y[ind1], error[ind1], self.alpha[ind1]):
                        ind2 = self.select_second_alpha(ind1, error)
                        change_pairs += self.update(ind1, ind2, error, x_dot)
            if entire_set:  # 当前是对整个数据集进行搜索，则下一次搜索间隔边界上的支持向量
                entire_set = False
            elif change_pairs == 0:
                entire_set = True  # 当前是对间隔边界上的支持向量进行搜索，若未发生任何改变，则下一次搜索整个数据集
            step += 1
        index = np.squeeze(np.argwhere(self.alpha > 0)) #选出alpha非零的样本
        self.x = self.x[index]
        self.y = self.y[index]
        self.alpha = self.alpha[index]

    def predict(self, x):
        """
        预测x的类别
        :param x: 传入二维数组，默认是传入多个样本
        :return:
        """
        kernel = self._dot_function(x, self.x)
        return np.sign(np.sum(self.y * self.alpha * kernel, axis=1) + self.b)

    def predict_prob(self, x):
        """
        预测样本距离超平面的距离
        :param x: 传入二维数组，默认是传入多个样本
        :return:
        """
        kernel = self._dot_function(x, self.x)
        return np.sum(self.y * self.alpha * kernel, axis=1) + self.b

