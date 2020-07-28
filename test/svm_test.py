from ml.classfication import SVM
import numpy as np

if __name__ == "__main__":
    from sklearn.datasets import load_digits

    data = load_digits(n_class=2)
    x_data = data['data']
    y_data = data['target']
    inds = np.where(y_data == 0)[0]
    y_data[inds] = -1
    print(len(y_data))
    np.random.seed(1203)
    split_label = np.random.random(y_data.shape[0]) < 0.2
    x_train, y_train, x_test, y_test = x_data[split_label], y_data[split_label], x_data[~ split_label], y_data[
        ~split_label]
    clf = SVM(epsilon=1e-6, C=1.0, kernel='rbf', kernel_params={"gamma": 0.0001})
    clf.fit(x_train, y_train)
    print("acc_num", np.sum(clf.predict(x_test) == y_test) / y_test.shape[0])
    print(len(y_test))
