# coding: utf-8
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# np.array = メトリクス生成
# np.zeros = 0メトリクス生成
# np.dot = *行列掛け算
# np.subtract = -行列引き算
# npobj * 2 = 要素単位の演算




# --- 単特徴の線形回帰 ---
# ある都市に出店するか否かの判断する為
# 人口(x)から利益(y)を予測する

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
m = sum(1 for line in data)

# 定数項(インターセプト)を入れた特徴ベクトル(人口)
# X = [ones(m, 1), data(:,1)];
X = np.array(map(lambda xi: [float(1), xi], X))

# xとyの
# theta = zeros(2, 1);
theta = np.zeros(2)

# 最急降下ループ数
# iterations = 1500;
iterations = 1500

# 学習率
# alpha = 0.01;
alpha = 0.01

# % コストの計算
# % # J = (1 / (2 * m)) * ((h(xi) - yi)^2)
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
plt.plot(X[:,1], np.dot(X, theta), label='Training data')
plt.legend()
plt.show()

# predict1 = [1, 3.5] *theta;
# fprintf('For population = 35,000, we predict a profit of %f\n',...predict1*10000);
# predict2 = [1, 7] * theta;
# fprintf('For population = 70,000, we predict a profit of %f\n',...predict2*10000);
#
# theta0_vals = linspace(-10, 10, 100);
# theta1_vals = linspace(-1, 4, 100);
#
# J_vals = zeros(length(theta0_vals), length(theta1_vals));
#
# for i = 1:length(theta0_vals)
#   for j = 1:length(theta1_vals)
#   t = [theta0_vals(i); theta1_vals(j)];
#   J_vals(i,j) = computeCost(X, y, t);
#   end
# end
#
# J_vals = J_vals';
# figure;
# surf(theta0_vals, theta1_vals, J_vals)
# xlabel('\theta_0');
# ylabel('\theta_1');
#
# figure;
# contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# xlabel('\theta_0'); ylabel('\theta_1');
# plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
