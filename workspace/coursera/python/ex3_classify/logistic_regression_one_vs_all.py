from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
import matplotlib.cm as cm
from scipy import optimize
from scipy.io import loadmat

# --- ロジスティック回帰、one vs all ---
# 手書き文字画像の分類を行う。

def display_data(X):
    """
    ベクトル化した画像を特徴をimshowで表示
    正方形画像にしか対応しない
    """
    m, n = X.shape
    example_width = int(np.around(np.sqrt(n)))
    example_height = int(n / example_width)
    display_rows = int(np.sqrt(m))
    display_cols = int(m / display_rows)
    display_array = np.ones((
        display_rows * example_height, display_cols * example_width
    ))
    for i in range(display_rows):
        for j in range(display_cols):
            idx = i * display_cols + j
            image_part = X[idx, :].reshape((example_height, example_width))
            display_array[
                (j * example_height):((j + 1) * example_height),
                (i * example_width):((i + 1) * example_width)
            ] = image_part
    plot.imshow(display_array.T, cm.Greys)
    plot.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cost_function(theta, X, y):
    """
    コスト関数(ロジスティック回帰)
    """
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    cost = sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    grad = X.T.dot(h - y)
    return (cost / m, grad / m)


def cost_function_reg(theta, X, y, lambda_):
    """
    コスト関数(正規化されたロジスティック回帰)
    """
    m = X.shape[0]
    cost, gradient = cost_function(theta, X, y)
    reg_cost = (lambda_ / (2.0 * m)) * np.sum(theta[1:] ** 2)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0
    return cost + reg_cost, gradient + reg_gradient


def one_vs_all(X, y, num_labels, _lambda, batch_count=50):
    """
    one vs all アルゴリズム (対象の分類とそれ以外で2値分類する)
    """
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n))
    for c in range(1, num_labels + 1):
        initial_theta = np.zeros(n)
        target = np.vectorize(int)(y == c)
        result = optimize.minimize(
            cost_function_reg,
            initial_theta,
            args=(X, target, _lambda),
            method='CG',
            jac=True,
            options={
                'maxiter': batch_count,
                'disp': False,
            }
        )
        theta = result.x
        cost = result.fun
        print('Training theta for label %d | cost: %f' % (c, cost))
        all_theta[c - 1, :] = theta
    return all_theta


def predict_one_vs_all(theta, X):
    """
    one vs all のlabel予測を返す
    labelは0から始まる為1を足す
    """
    return 1 + np.argmax(sigmoid(X.dot(theta.T)), axis=1)

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
    num_labels = len(list(set(y)))

    # .matファイルのサンプルからランダムに100個可視化
    sel = np.random.permutation(X)[:100]
    display_data(sel)

    # インターセプト項の挿入
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # トレーニング
    #   λ(ラムダ) = 正規化パラメータ
    #     if lambda > 1?: ペナルティー上昇、0に近づき直線化、アンダーフィッティング
    #     if lambda < 1?: ペナルティー下降、1に近づき曲線化、オーバーフィッティング
    _lambda = 1
    # トレーニング回数
    batch_count = 50
    all_theta = one_vs_all(X, y, num_labels, _lambda, batch_count=batch_count)

    # 予測の実行
    predictions = predict_one_vs_all(all_theta, X)

    # 精度の出力
    # TODO: トレーニングデータとテストデータは分けるべき。
    accuracy = 100 * np.mean(predictions == y)
    print('accuracy(精度): %0.2f %%' % accuracy)
