from ml.linear_model import LogisticRegression
import numpy as np

if __name__ == '__main__':
    from sklearn import datasets
    data = datasets.load_digits(n_class=2)
    x_data = data['data']
    y_data = data['target']
    inds = np.where(y_data == 0)[0]
    y_data[inds] = -1
    split_label = np.random.random(y_data.shape[0]) < 0.1
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[~ split_label]
    clf = LogisticRegression(l2=0.1)
    clf.fit(x_train, y_train)
    acc_num = 0
    print(np.sum(clf.predict(x_test) == y_test)/len(y_test))
    print(np.sum(np.abs(clf.w))+np.abs(clf.b), acc_num/len(y_test))

