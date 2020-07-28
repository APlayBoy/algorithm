from ml.tree import CART
import numpy as np

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x_data = load_iris().data
    y_data = load_iris().target

    split_label = np.random.random(y_data.shape[0]) < 0.5
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~ split_label]
    clf = CART(min_sample=5)
    clf.fit(x_train, y_train, [0, 1, 2, 3])
    print("classfication acc", np.sum(clf.predict(x_test) == y_test) / y_test.shape[0])

    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    x_data, y_data = diabetes["data"], diabetes["target"]

    score_list = []
    for n in range(20):
        split_label = np.random.random(y_data.shape[0]) < 0.8
        x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
            ~ split_label]
        clf = CART(min_sample=8, classfication=False) #回归问题比分类问题的叶子节点要多
        clf.fit(x_train, y_train, continuous=list(range(x_train.shape[1])))
        predict = clf.predict(x_test)
        print("regresion score ", n, 1 - np.sum(np.square(y_test - predict)) / np.sum(np.square(y_test - np.mean(y_test))))
        score_list.append(1 - np.sum(np.square(y_test - predict)) / np.sum(np.square(y_test - np.mean(y_test))))
    print("avg score", sum(score_list)/len(score_list))
