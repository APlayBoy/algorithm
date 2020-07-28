from ml.ensemble import RandomForest
import numpy as np

if __name__ == '__main__':

    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    x_data, y_data = diabetes["data"], diabetes["target"]

    split_label = np.random.random(y_data.shape[0]) < 0.8
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~ split_label]
    clf = RandomForest(num_tree=100, num_feature=3, classfication=False)
    clf.fit(x_train, y_train, tree_params=
    {"continuous":list(range(x_train.shape[1])), "max_depth":3, "min_sample":3})
    predict = clf.predict(x_test)
    print("regresion score ", 1 - np.sum(np.square(y_test - predict)) / np.sum(np.square(y_test - np.mean(y_test))))

    from sklearn import datasets
    data = datasets.load_digits(n_class=2)
    x_data = data['data']
    y_data = data['target']
    inds = np.where(y_data == 0)[0]
    y_data[inds] = -1
    split_label = np.random.random(y_data.shape[0]) < 0.5
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~ split_label]
    clf = RandomForest(num_tree=100, num_feature=3, ri=False, L=3, classfication=True)
    clf.fit(x_train, y_train, tree_params=
    {"continuous":list(range(x_train.shape[1])), "max_depth": 3, "min_sample": 3})
    print("classfication acc: ", np.sum(clf.predict(x_test) == y_test) / len(y_test))
