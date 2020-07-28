from ml.linear_model import SoftMaxRegression
from sklearn.datasets import load_digits
import numpy as np


if __name__ == '__main__':
    data = load_digits()
    x_data = data['data']
    y_data = data['target']
    split_label = np.random.random(y_data.shape[0]) < 0.5
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[~split_label]
    clf = SoftMaxRegression(alpha=0.1, l2=1e-3)
    clf.fit(x_train, y_train)
    print(np.sum(clf.predict(x_test) == y_test) / len(y_test))
