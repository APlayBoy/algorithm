"""
实现线性回归, 支持l2正则
w = (inv(X.T@X+alpha*np.eye(X.shape[1]))@X.T@y).T
"""
import numpy as np


class LinearRegression:

    def __init__(self, alpha=None):
        self.w = None
        self.alpha = alpha

    def fit(self, x, y):
        fit_data = np.insert(x, x.shape[1], 1, axis=1)
        if self.alpha is not None and self.alpha != 0:
            self.w = (np.linalg.inv(fit_data.T @ fit_data + self.alpha * np.eye(fit_data.shape[1])) @ fit_data.T @ y).T
        else:
            self.w = (np.linalg.inv(fit_data.T @ fit_data) @ fit_data.T @ y).T

    def predict(self, x):
        predict_data = np.insert(x, x.shape[1], 1, axis=1)
        return self.w @ predict_data.T

    def score(self, x, y):
        predict_y = self.predict(x)
        return 1 - np.sum(np.square(y - predict_y)) / np.sum(np.square(y - np.mean(y)))



