import numpy as np
from ml.linear_model import LinearRegression
from sklearn import datasets

if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    x_data, y_data = diabetes["data"], diabetes["target"]
    split_label = np.random.random(y_data.shape[0]) < 0.8
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~ split_label]
    clf = LinearRegression(alpha=0.1)
    clf.fit(x_train, y_train)
    for i, j in zip(clf.predict(x_test), y_test):
        print(i, j)
    print(clf.score(x_test, y_test))

