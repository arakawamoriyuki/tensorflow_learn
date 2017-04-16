# coding: utf-8
from __future__ import print_function

import numpy as np
from numpy.linalg import pinv
from matplotlib import pyplot as plot

from gradient_descent_single import gradient_descent

# --- 多変量の線形回帰(多特徴) ---
# 家を売った際の値段を予測する為
# 広さ(x1)と部屋数(x2)から売値(y)を予測する

def normalize_features(X, mu=None, sigma=None):
    m = X.shape[0]
    Xnorm = np.zeros_like(X)
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    # don't change the intercept term
    mu[0] = 0.0
    sigma[0] = 1.0
    for i in range(m):
        Xnorm[i, :] = (X[i, :] - mu) / sigma
    return Xnorm, mu, sigma


if __name__ == '__main__':
    # データのロード、変数の初期化
    data = np.loadtxt('../../machine-learning-ex1/ex1/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = X.shape[0]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    alpha = 0.01
    num_iters = 400
    theta = np.zeros(3)

    # フィーチャースケーリング(feature scaling)と平均正則化(mean normalization)
    Xnorm, mu, sigma = normalize_features(X)

    # 最急降下法
    [theta, J_history] = gradient_descent(Xnorm, y, theta, alpha, num_iters)

    # 収束の可視化
    plot.plot(J_history, '-b')
    plot.xlabel('Number of iterations')
    plot.ylabel('Cost J')
    plot.show()

    # 予測値の出力
    size = 1650
    rooms = 3
    x = np.array([[1.0, size, rooms]])
    x, _, _ = normalize_features(x, mu, sigma)
    price = x.dot(theta)[0]
    print("広さ{size}で{rooms}部屋の家売値予測値は{price}です。".format(
        size=size,
        rooms=rooms,
        price=price
    ))
