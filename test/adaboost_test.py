from sklearn.datasets import load_digits
from ml.linear_model import LogisticRegression
from ml.boosting import AdaBoost
import numpy as np

if __name__ == '__main__':
    data = load_digits(n_class=2)
    x_data = data['data']
    y_data = data['target']
    inds = np.where(y_data == 0)[0]
    y_data[inds] = -1
    np.random.seed(1203)
    split_label = np.random.random(y_data.shape[0]) < 0.5
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~split_label]
    print("train num:", y_train.shape[0], "test num:", y_test.shape[0])
    base = LogisticRegression()
    max_step = 6
    base.fit(x_train, y_train, max_step=3)
    print("base_acc_num", np.sum(base.predict(x_test) == y_test) / y_test.shape[0])
    clf = AdaBoost(0.02)

    clf.fit(x_train, y_train, LogisticRegression, classfier_params={'max_step': max_step})
    print("ada_base_classfier num", len(clf.alpha))
    print("ada_acc_num", np.sum(clf.predict(x_test) == y_test) / y_test.shape[0])
