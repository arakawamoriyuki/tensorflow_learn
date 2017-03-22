# coding: utf-8
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SHOW_PLOT = False

# --- 分類 ---
# 過去の2つの試験結果データから学生が大学に入学するかどうかを予測


# ロードと変数初期化
# data = load('ex2data1.txt');
data = []
for line in open('../../machine-learning-ex2/ex2/ex2data1.txt', 'r'):
    splits = line.strip().split(',')
    x1 = splits[0]
    x2 = splits[1]
    y = splits[2]
    data.append([x1, x2, y])

# 特徴ベクトル(試験の点数x2)
# X = data(:, 1:2);
X = np.array(map(lambda d: [float(d[0]), float(d[1])], data))

# 答えベクトル(入学フラグ)
# y = data(:, 3);
y = np.array(map(lambda d: float(d[2]), data))

# プロット
# function plotData(X, y)
#     figure; hold on;
#     pos = find(y == 1);
#     neg = find(y == 0);
#     plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
#     plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
#     hold off;
# end
# plotData(X, y);
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
def plot_data(X, y, xlabel='x', ylabel='y'):
    positive = np.nonzero(y == 1)
    negative = np.nonzero(y == 0)
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label='positive')
    plt.plot(X[negative, 0], X[negative, 1], 'ko', label='positive')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
if SHOW_PLOT:
    plot_data(X, y, xlabel='Exam 1 score', ylabel='Exam 2 score')

# データセットの数(m)、特徴の数(n)
# [m, n] = size(X);
m, n = X.shape

# 定数項(インターセプト)を入れた特徴メトリクス(人口)
# X = [ones(m, 1) X];
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# パラメータの初期化
# initial_theta = zeros(n + 1, 1);
initial_theta = np.zeros((n + 1, 1))

# コスト(cost)と勾配(gradient)を計算
# function g = sigmoid(z)
#     g = zeros(size(z));
#     g = ones(size(z)) ./ (1.0 + exp(-z));
# end
# function [J, grad] = costFunction(theta, X, y)
#     m = length(y);
#     J = 0;
#     grad = zeros(size(theta));
#     h = sigmoid(X * theta);
#     cost = sum(-y .* log(h) - (1 - y) .* log(1 - h));
#     grad = X' * (h - y);
#     J = cost / m;
#     grad = grad / m;
# end
# [cost, grad] = costFunction(initial_theta, X, y);
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def cost_function(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    cost = sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    grad = X.T.dot(h - y)
    return (cost / m, grad / m)
cost, gradient = cost_function(initial_theta, X, y)

print(gradient)

# fprintf('Cost at initial theta (zeros): %f\n', cost);
# fprintf('Gradient at initial theta (zeros): \n');
# fprintf(' %f \n', grad);
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ============= Part 3: Optimizing using fminunc  =============
# %  In this exercise, you will use a built-in function (fminunc) to find the
# %  optimal parameters theta.
#
# %  Set options for fminunc
# options = optimset('GradObj', 'on', 'MaxIter', 400);
#
# %  Run fminunc to obtain the optimal theta
# %  This function will return theta and the cost
# [theta, cost] = ...
# 	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
#
# % Print theta to screen
# fprintf('Cost at theta found by fminunc: %f\n', cost);
# fprintf('theta: \n');
# fprintf(' %f \n', theta);
#
# % Plot Boundary
# plotDecisionBoundary(theta, X, y);
#
# % Put some labels
# hold on;
# % Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
#
# % Specified in plot order
# legend('Admitted', 'Not admitted')
# hold off;
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of
# %  our model.
# %
# %  Your task is to complete the code in predict.m
#
# %  Predict probability for a student with score 45 on exam 1
# %  and score 85 on exam 2
#
# prob = sigmoid([1 45 85] * theta);
# fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
#          'probability of %f\n\n'], prob);
#
# % Compute accuracy on our training set
# p = predict(theta, X);
#
# fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;

