# coding: utf-8
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- 多特徴の線形回帰 ---
# 家を売った際の値段を予測する為
# 広さ(x1)と部屋数(x2)から売値(y)を予測する

# ロードと変数初期化
# data = load('ex1data2.txt');
# data = load('./machine-learning-ex1/ex1/ex1data2.txt');
data = []
for line in open('../../machine-learning-ex1/ex1/ex1data2.txt', 'r'):
    splits = line.strip().split(',')
    x1 = splits[0]
    x2 = splits[1]
    y = splits[2]
    data.append([x1, x2, y])

# 特徴ベクトル(広さと部屋数)
# X = data(:, 1:2);
X = np.array(map(lambda d: [float(d[0]), float(d[1])], data))

# 答えベクトル(売値)
# y = data(:, 3);
y = np.array(map(lambda d: float(d[2]), data))

# データセットの数
# m = length(y);
m = len(y)

# フィーチャースケーリング(feature scaling)と平均正則化(mean normalization)
# function [X_norm, mu, sigma] = featureNormalize(X)
#     X_norm = X;
#     numColumns = size(X, 2);
#     mu = mean(X);
#     sigma = std(X);
#     for i = 1:numColumns
#         X_norm(:,i) = (X(:, i) - mu(i)) / sigma(i);
#     end;
# end
# [X mu sigma] = featureNormalize(X);
def feature_normalize(X):
    X_norm = X
    num_columns = X.shape[1]    # column size
    mu = np.mean(X, axis=0)     # column毎の平均
    sigma = np.std(X, axis=0)   # 標準偏差
    for i in range(num_columns):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]
    return (X_norm, mu, sigma)
(X, mu, sigma) = feature_normalize(X)

# 定数項(インターセプト)を入れた特徴メトリクス(人口)
# X = [ones(m, 1) X];
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# 学習率
# alpha = 0.01;
alpha = 0.01

# 最急降下ループ数
# num_iters = 400;
num_iters = 400

# 傾きパラメータの初期化
# theta = zeros(3, 1);
theta = np.zeros(3)

# 複数特徴のコストの計算
# function J = computeCostMulti(X, y, theta)
#     m = length(y);
#     J = 0;
#     cost = 0;
#     for i = 1:m
#       cost = cost + (theta' * X(i,:)' - y(i))^2;
#     end;
#     J = cost / (2 * m);
# end
def compute_cost_multi(X, y, theta):
    m = len(y)
    J = 0
    cost = 0
    for i in range(m):
        cost = cost + ((np.dot(theta.transpose(), X[i, :].transpose()) - y[i]) ** 2)
    J = cost / (2 * m)
    return J

# 複数特徴の最急降下法
# function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
#     m = length(y);
#     J_history = zeros(num_iters, 1);
#     for iter = 1:num_iters
#         h = X * theta;
#         errors = h - y;
#         delta = X' * errors;
#         theta = theta - (alpha / m) * delta;
#         J_history(iter) = computeCostMulti(X, y, theta);
#     end
# end
# [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = np.dot(X, theta)
        errors = h - y
        delta = np.dot(X.transpose(), errors)
        theta = theta - (alpha / m) * delta
        J_history[i] = compute_cost_multi(X, y, theta)
    return (theta, J_history)
(theta, J_history) = gradient_descent_multi(X, y, theta, alpha, num_iters)

# パラメータthetaの収束グラフ(convergence graph)
# figure;
# plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
# xlabel('Number of iterations');
# ylabel('Cost J');
plt.plot(range(1, J_history.size+1), J_history)
plt.show()

# 売値の予測
# size = 1650;
# rooms = 3;
# price = [1 (size-mu(1))/sigma(1) (rooms-mu(2))/sigma(2)]*theta;
size = 1650
rooms = 3
x1 = (size - mu[0]) / sigma[0]
x2 = (rooms - mu[1]) / sigma[1]
price = np.dot(np.array([1, x1, x2]), theta)
print("最急降下法: 広さ{size}で{rooms}部屋の家売値予測値は{price}です。".format(
    size=size,
    rooms=rooms,
    price=price
))


# --- 正規方程式での計算方法(Normal equation) ---

data = []
for line in open('../../machine-learning-ex1/ex1/ex1data2.txt', 'r'):
    splits = line.strip().split(',')
    x1 = splits[0]
    x2 = splits[1]
    y = splits[2]
    data.append([x1, x2, y])
X = np.array(map(lambda d: [float(d[0]), float(d[1])], data))
y = np.array(map(lambda d: float(d[2]), data))
m = len(y)
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# 正規方程式
# function [theta] = normalEqn(X, y)
#     theta = zeros(size(X, 2), 1);
#     theta = pinv(X' * X) * X' * y;
# end
# theta = normalEqn(X, y);
def normal_eqn(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)
    return theta
theta = normal_eqn(X, y)

# 売値の予測
# size = 1650;
# rooms = 3;
# x = [1 size rooms]';
# price = theta' * x;
size = 1650
rooms = 3
x = np.array([1, size, rooms]).transpose()
price = np.dot(theta.transpose(), x)
print("正規方程式: 広さ{size}で{rooms}部屋の家売値予測値は{price}です。".format(
    size=size,
    rooms=rooms,
    price=price
))
