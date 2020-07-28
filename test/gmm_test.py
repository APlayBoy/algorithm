from ml.cluster import GMM
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np


# 生成数据
def generate_data(true_mu, true_var):
    # 第一簇的数据
    nums = [400, 600, 1000]
    gen_funcs = [np.random.multivariate_normal, np.random.multivariate_normal, np.random.multivariate_normal]
    x_data = [f(mu, np.diag(var), n) for n, f, mu, var in zip(nums, gen_funcs, true_mu, true_var)]
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(x_data[0][:, 0], x_data[0][:, 1], s=5)
    plt.scatter(x_data[1][:, 0], x_data[1][:, 1], s=5)
    plt.scatter(x_data[2][:, 0], x_data[2][:, 1], s=5)
    plt.show()
    return np.vstack(x_data)

# 画出聚类图像
def plot_clusters(data, mus, vars, mus_true=None, var_true=None):
    colors = ['b', 'g', 'r']
    n_clusters = len(mus)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(data[:, 0], data[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(mus[i], 3 * vars[i][0], 3 * vars[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (mus_true is not None) & (var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(mus_true[i], 3 * var_true[i][0], 3 * var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()


if __name__ == '__main__':
    # 生成数据
    generate_mus = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    generate_vars = [[1, 3], [2, 2], [6, 2]]
    generate_x = generate_data(generate_mus, generate_vars)
    gmm = GMM(3)
    gmm.fit(generate_x, step=50)
    plot_clusters(generate_x, gmm.mus, gmm.vars, generate_mus, generate_vars)
    print(gmm.predict(generate_x[0]))
