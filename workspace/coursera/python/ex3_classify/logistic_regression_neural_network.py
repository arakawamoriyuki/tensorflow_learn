from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat

from logistic_regression_one_vs_all import display_data, sigmoid


def predict(Theta1, Theta2, X):
    """
    Neural Networkの予測を返す
    """
    m = X.shape[0]
    A2 = sigmoid(X.dot(Theta1.T))
    A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)
    A3 = sigmoid(A2.dot(Theta2.T))
    predictions = 1 + np.argmax(A3, axis=1)
    return predictions


if __name__ == '__main__':
    # データのロード、変数の初期化
    # .mat =
    #   Microsoft Access table shortcut file?
    #   画像のグレースケールpixel値の2次配列
    #   サンプル数 = 5000
    #   分類数 = 10(0 ~ 9)
    #   1画像 = 20x20 = 400pixel = 特徴数400
    data = loadmat('../../machine-learning-ex3/ex3/ex3data1.mat')
    X = data['X']
    y = data['y'].flatten()
    m = X.shape[0]

    # .matファイルのサンプルからランダムに100個可視化
    sel = np.random.permutation(X)[:100]
    display_data(sel)

    # インターセプト項の挿入
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # 保存している97%認識率のNeural Networkパラメータをロードする
    # layer = 入力層(400),中間層(400),出力層(2)
    weights = loadmat('../../machine-learning-ex3/ex3/ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    # 精度の出力
    predictions = predict(Theta1, Theta2, X)
    accuracy = 100 * np.mean(predictions == y)
    print('accuracy(精度): %0.2f %%' % accuracy)

    # TODO: zipで回して答えも表示
    # random_X = np.random.permutation(X)
    # for i in range(m):
    #     example = random_X[i].reshape(1, -1)
    #     prediction = predict(Theta1, Theta2, example)
    #     print('予測: %d' % (prediction % 10))
    #     display_data(example[:, 1:])

    # 予測の出力
    for _x, label in zip(X, y):
        prediction = predict(Theta1, Theta2, np.array([_x]))[0]
        print('予測: %d , 答え: %d' % (prediction % 10, label))
        display_data(_x)
