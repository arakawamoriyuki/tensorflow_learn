# coding: utf-8
from __future__ import print_function

import numpy as np
from numpy.linalg import pinv
from matplotlib import pyplot as plot

from gradient_descent_single import gradient_descent

# --- 正規方程式(多特徴) ---
# 家を売った際の値段を予測する為
# 広さ(x1)と部屋数(x2)から売値(y)を予測する
#
# - 特徴数がだいたい10000以下なら早い
# - 学習率が必要ない

def normal_equation(X, y):
    return pinv(X.T.dot(X)).dot(X.T).dot(y)


if __name__ == '__main__':
    # データのロード、変数の初期化
    data = np.loadtxt('../../machine-learning-ex1/ex1/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = X.shape[0]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # 正規方程式(Normal equation) だいたい10000特徴以下なら早い
    theta = normal_equation(X, y)
    size = 1650
    rooms = 3
    x = np.array([[1.0, size, rooms]])
    price = x.dot(theta)[0]

    print("広さ{size}で{rooms}部屋の家売値予測値は{price}です。".format(
        size=size,
        rooms=rooms,
        price=price
    ))
