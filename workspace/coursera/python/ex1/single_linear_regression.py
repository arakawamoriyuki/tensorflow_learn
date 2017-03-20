# coding: utf-8
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 単特徴の線形回帰 ---
# ある都市に出店するか否かの判断する為
# 人口(x)から利益(y)を予測する
# データセットはそれぞれ10000で割ったfloat。x/=10000 y/=10000

# ロードと変数初期化
# data = load('ex1data1.txt');
# data = load('./machine-learning-ex1/ex1/ex1data1.txt');
data = []
for line in open('./machine-learning-ex1/ex1/ex1data1.txt', 'r'):
    splits = line.strip().split(',')
    x = splits[0]
    y = splits[1]
    data.append([x, y])

# 特徴ベクトル(人口)
# X = data(:, 1);
X = np.array(map(lambda d: float(d[0]), data))

# 答えベクトル(利益)
# y = data(:, 2);
y = np.array(map(lambda d: float(d[1]), data))

# データセットの数
# m = length(y);
m = len(y)

# 定数項(インターセプト)を入れた特徴メトリクス(人口)
# X = [ones(m, 1), data(:,1)];
X = np.array(map(lambda xi: [float(1), xi], X))

# 傾きパラメータの初期化
# theta = zeros(2, 1);
theta = np.zeros(2)

# 最急降下ループ数
# iterations = 1500;
iterations = 1500

# 学習率
# alpha = 0.01;
alpha = 0.01

# コストの計算
# function J = computeCost(X, y, theta)
#   m = length(y);
#   costs = ((X * theta) - y) .^ 2;
#   J = sum(costs) / (2 * m);
# end
def compute_cost(X, y, theta):
    m = len(y)
    costs = np.subtract(np.dot(X, theta), y) ** 2
    J = np.sum(costs) / (2 * m)
    return J

# % 最急降下法
# % # repeat until convergence {
# % #   Θj := Θj - (α * (∂ / ∂Θj)) J(θ0, θ1)   (for j = 0 and j = 1)
# % # }
# function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
#   m = length(y);
#   J_history = zeros(num_iters, 1);
#   for iter = 1:num_iters
#     h = X * theta;
#     errors = h - y;
#     delta = X' * errors;
#     theta = theta - (alpha / m) * delta;
#     J_history(iter) = computeCost(X, y, theta);
#   end
# end
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        h = np.dot(X, theta)
        errors = h - y
        delta = np.dot(X.transpose(), errors)
        theta = theta - (alpha / m) * delta
        J_history[iter] = compute_cost(X, y, theta)
    return (theta, J_history)

# 最急降下法でパラメータの計算
# theta = gradientDescent(X, y, theta, alpha, iterations);
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# プロット表示
# plot(X(:,2), X*theta, '-')
# legend('Training data', 'Linear regression')
plt.plot(X[:,1], y, 'ro', label='Training data')
plt.plot(X[:,1], np.dot(X, theta), label='Linear regression')
plt.legend()
plt.show()

# 予測値の出力
# predict1 = [1, 3.5] *theta;
# fprintf('For population = 35,000, we predict a profit of %f\n',...predict1*10000);
# predict2 = [1, 7] * theta;
# fprintf('For population = 70,000, we predict a profit of %f\n',...predict2*10000);
population = 3.5
predict = np.dot([1, population], theta)
print("人口{population}人の都市で出店した場合の利益予測値は{predict}です。".format(population=population*10000, predict=predict*10000))
population = 7.0
predict = np.dot([1, population], theta)
print("人口{population}人の都市で出店した場合の利益予測値は{predict}です。".format(population=population*10000, predict=predict*10000))


# --- Jの可視化 ---

# theta0_vals = linspace(-10, 10, 100);
# theta1_vals = linspace(-1, 4, 100);
theta0_vals = np.linspace(-10, 10, num=100)
theta1_vals = np.linspace(-1, 4, num=100)

# J_vals = zeros(length(theta0_vals), length(theta1_vals));
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# for i = 1:length(theta0_vals)
#   for j = 1:length(theta1_vals)
#     t = [theta0_vals(i); theta1_vals(j)];
#     J_vals(i,j) = computeCost(X, y, t);
#   end
# end
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t);

# J_vals = J_vals';
J_vals = J_vals.transpose()

# figure;
# surf(theta0_vals, theta1_vals, J_vals)
# xlabel('\theta_0');
# ylabel('\theta_1');
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1)
plt.show()

# figure;
# contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# xlabel('\theta_0'); ylabel('\theta_1');
# plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, num=20))
plt.plot(theta[0], theta[1], 'rx')
plt.show()
