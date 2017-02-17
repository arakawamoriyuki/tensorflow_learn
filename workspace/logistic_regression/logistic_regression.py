from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import rand, multivariate_normal
import matplotlib.pyplot as plt

# データセット
class Dataset():
  def __init__(self):
    variance = 40
    n1, n2 = 60, 40
    mu1, mu2 = [13,10], [0,-3]
    cov1 = np.array([[variance,0],[0,variance]])
    cov2 = np.array([[variance,0],[0,variance]])
    data1 = multivariate_normal(mu1,cov1,n1)
    data2 = multivariate_normal(mu2,cov2,n2)
    data = np.r_[np.c_[data1,np.ones(n1)], np.c_[data2,np.zeros(n2)]]
    np.random.shuffle(data)

    self.test_data, self.test_label = np.hsplit(data[0:20],[2])
    self.train_data, self.train_label = np.hsplit(data[20:],[2])
    self.index = 0

  def next_batch(self, n):
    if self.index + n > len(self.train_data):
        self.index = 0
    data = self.train_data[self.index:self.index+n]
    label = self.train_label[self.index:self.index+n]
    self.index += n
    return data, label

dataset = Dataset()
mult = dataset.train_data.flatten().mean()

batch_xs, batch_ys = dataset.next_batch(10)
feed = {x: dataset.test_data, y_: dataset.test_label}


def plot_result(dataset, weight, bias, mult):
  fig = plt.figure()
  subplot = fig.add_subplot(1,1,1)

  data0_x, data0_y, data1_x, data1_y = [], [], [], []
  for i in range(len(dataset.train_data)):
    if dataset.train_label[i][0] == 0:
      data0_x.append(dataset.train_data[i][0])
      data0_y.append(dataset.train_data[i][1])
    else:
      data1_x.append(dataset.train_data[i][0])
      data1_y.append(dataset.train_data[i][1])
  subplot.scatter(data0_x, data0_y, marker='x', color='blue')
  subplot.scatter(data1_x, data1_y, marker='o', color='blue')

  data0_x, data0_y, data1_x, data1_y = [], [], [], []
  for i in range(len(dataset.test_data)):
    if dataset.test_label[i][0] == 0:
      data0_x.append(dataset.test_data[i][0])
      data0_y.append(dataset.test_data[i][1])
    else:
      data1_x.append(dataset.test_data[i][0])
      data1_y.append(dataset.test_data[i][1])
  subplot.scatter(data0_x, data0_y, marker='x', color='red')
  subplot.scatter(data1_x, data1_y, marker='o', color='red')

  xs, ys = np.hsplit(dataset.train_data,[1])
  wx, wy, b = weight[0][0], weight[1][0], bias[0]
  linex = np.arange(xs.min()-5, xs.max()+5)
  liney = - linex * wx/wy - b*mult/wy
  subplot.plot(linex, liney, color='red')
  plt.show()

# Main
if __name__ == '__main__':

  dataset = Dataset()
  sess = tf.InteractiveSession()
  writer = tf.train.SummaryWriter('/tmp/logistic_logs', sess.graph_def)
  mult = dataset.train_data.flatten().mean()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 2])
  w = tf.Variable(tf.zeros([2, 1]))
  b = tf.Variable(tf.zeros([1]))
  y = tf.sigmoid(tf.matmul(x, w) + b*mult)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])
  log_probability = tf.reduce_sum(y_*tf.log(y) + (1-y_)*tf.log(1-y))
  train_step = tf.train.GradientDescentOptimizer(0.001).minimize(-log_probability)
  correct_prediction = tf.equal(tf.sign(y-0.5), tf.sign(y_-0.5))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Logging data for TensorBoard
  _ = tf.histogram_summary('weight', w)
  _ = tf.histogram_summary('bias', b)
  _ = tf.histogram_summary('probability of training data', y)
  _ = tf.scalar_summary('log_probability', log_probability)
  _ = tf.scalar_summary('accuracy', accuracy)

  # Train
  tf.initialize_all_variables().run()

  for i in range(240):
    batch_xs, batch_ys = dataset.next_batch(10)
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)

    feed = {x: dataset.test_data, y_: dataset.test_label}
    summary_str, lp, acc = sess.run(
      [tf.merge_all_summaries(), log_probability, accuracy], feed_dict=feed)
    writer.add_summary(summary_str, i)
    print('LogProbability and Accuracy at step %s: %s, %s' % (i, lp, acc))

  plot_result(dataset, w.eval(), b.eval(), mult)