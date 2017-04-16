# coding: utf-8
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy import optimize

# --- ロジスティック回帰 ---
# 学生が大学に入学できるかどうかを予測する。
# 試験結果1(x1)と試験結果2(x2)から入学できるか(y)を予測する。

def plot_data(X, y, show=True):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    plot.xlabel('Exam 1 score')
    plot.ylabel('Exam 2 score')
    if show:
        plot.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cost_function(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    cost = sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    grad = X.T.dot(h - y)
    return (cost / m, grad / m)


def predict(theta, X):
    return sigmoid(X.dot(theta)) >= 0.5


if __name__ == '__main__':
    # データのロード、変数の初期化
    data1 = np.loadtxt('../../machine-learning-ex2/ex2/ex2data1.txt', delimiter=',')
    X = data1[:, 0:2]
    y = data1[:, 2]
    m, n = X.shape
    initial_theta = np.zeros(n + 1)

    # トレーニングデータの可視化
    plot_data(X, y)

    # インターセプト項の挿入
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # 1回で出したコストcostとそのパラメータを表示
    cost, grad = cost_function(initial_theta, X, y)
    print('Cost at initial theta (zeros): %f' % cost)
    print('Gradient at initial theta (zeros): \n %s' % grad)

    # scipy.optimizeを利用してcostの最小値を探す
    loop = 400
    wrapped = lambda t: cost_function(t, X, y)[0]
    result = optimize.minimize(
        wrapped,
        initial_theta,
        method='Nelder-Mead',
        options={
            'maxiter': loop,
            'disp': False,
        }
    )

    # scipy.optimizeを利用して指定回数ループして探した最小値のcostとそのパラメータを表示
    theta = result.x
    cost = result.fun
    print('Cost at theta found by scipy.optimize.minimize: %f' % cost)
    print('theta: \n %s' % theta)

    # 探したパラメータを元にトレーニングデータに決定境界を表示
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
    plot_data(X[:, 1:], y, show=False)
    plot.plot(plot_x, plot_y)
    plot.show()

    # 予測値の出力
    test1 = 45
    test2 = 85
    prob = sigmoid(np.array([1, test1, test2]).dot(theta))
    print('試験結果が{}点と{}点だった学生が入学できる確率は{}'.format(test1, test2, prob))

    # 精度の表示
    predictions = predict(theta, X)
    accuracy = 100 * np.mean(predictions == y)
    print('Train accuracy: %0.2f %%' % accuracy)
