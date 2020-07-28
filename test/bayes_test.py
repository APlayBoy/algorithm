import numpy as np
from ml.classfication import NBayes


if __name__ == '__main__':

    data = np.array([[0, 0, 1, -1], [1, 1, 2, -1], [1, 1, 2, 1], [1, 0, 0, 1],
                     [1, 0, 1, -1], [2, 0, 2, -1], [2, 1, 0, -1], [2, 1, 1, 1],
                     [2, 2, 2, 1], [2, 2, 1, 1], [3, 2, 1, 1], [3, 1, 0, 1],
                     [3, 1, 2, 1], [3, 2, 1, 1], [3, 2, 0, -1]])
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition)
    print(clf.predict(np.array([[2, 0, 1], [3, 1, 0], [2, 2, 2], [1, 1, 2]])))