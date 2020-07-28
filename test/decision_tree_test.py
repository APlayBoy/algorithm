from ml.tree import DecisionTree
import numpy as np


if __name__ == '__main__':
    print(np.argmax(np.array([2, 3 ,3, 4, 2])))
    data = np.array([['青年', '青年', '青年', '青年', '青年', '中年', '中年',
                      '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
                     ['否', '否', '是', '是', '否', '否', '否', '是', '否',
                      '否', '否', '否', '是', '是', '否'],
                     ['否', '否', '否', '是', '否', '否', '否', '是',
                      '是', '是', '是', '是', '否', '否', '否'],
                     ['一般', '好', '好', '一般', '一般', '一般', '好', '好',
                      '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
                     ['否', '否', '是', '是', '否', '否', '否', '是', '是',
                      '是', '是', '是', '是', '是', '否']])
    data = data.T
    x_data = data[:, :-1]
    y_data = data[:, -1]

    np.random.seed(1203)
    split_label = np.random.random(y_data.shape[0]) < 0.6
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~split_label]

    from sklearn.datasets import load_digits
    data = load_digits(n_class=2)
    x_data = data['data']
    y_data = data['target']
    np.random.seed(1203)
    split_label = np.random.random(y_data.shape[0]) < 0.8
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~split_label]
    print("train num:", y_train.shape[0], "test num:", y_test.shape[0])
    clf = DecisionTree()
    clf.fit(x_train, y_train)
    print("y_test: ", len(y_test))
    r = clf.predict(x_test) == y_test
    print(len(r), len(y_test))
    print(np.sum(r) / y_test.shape[0])
