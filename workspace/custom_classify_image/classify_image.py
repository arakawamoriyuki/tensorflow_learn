#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, itertools

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


# checkpoint path
CHECKPOINT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../models/classify_image/'

# tensorboard path
TENSORBOARD_PATH = '/tmp/tensorboard/classify_image/'


def inference(images_placeholder, keep_prob, num_classes=10, image_size=28, patch_size=5, output_channel=32, add_channel=2, layer=2):
    """ モデルを作成する関数
    @param
        images_placeholder      inputs()で作成した画像のplaceholder
        keep_prob               dropout率のplace_holder
        num_classes             分類数
        image_size              画像の1辺のpixel数
        patch_size              畳み込みの切り出しサイズ
        output_channel          最初の層のノード数
        add_channel             最初のノード数から以降乗算で増やす数
        layer                   層数
    @return
        cross_entropy       モデルの計算結果
    """

    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
        return tf.nn.conv2d(
            x,
            W,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

    # プーリング層の作成
    def max_pool_2x2(x):
        return tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    # 色数(カラーの場合rgbで3、モノクロは1)
    channel = 3

    # 入力をimage_size x image_size x 3に変形
    h_pool = tf.reshape(images_placeholder, [-1, image_size, image_size, channel])

    W_conv = None
    b_conv = None
    h_conv = None

    for layer_index in range(layer):
        # 畳み込み層
        # patch_sizeで画像を切り出し、元画像の全位置と比較し、一致率を算出
        with tf.name_scope('conv%d' % layer_index) as scope:
            if layer_index == 0:
                input_channel = channel
            else:
                input_channel = output_channel
                output_channel *= add_channel
            W_conv = weight_variable([patch_size, patch_size, input_channel, output_channel])
            b_conv = bias_variable([output_channel])
            h_conv = tf.nn.relu(conv2d(h_pool, W_conv) + b_conv)
        # プーリング層
        # 画像を重要な情報は残しつつ最大値を取って1/4に縮小する。
        with tf.name_scope('pool%d' % layer_index) as scope:
            h_pool = max_pool_2x2(h_conv)
        # print('W_conv' + str(layer_index) + ' = ' + str(W_conv))
        # print('b_conv' + str(layer_index) + ' = ' + str(b_conv))
        # print('h_conv' + str(layer_index) + ' = ' + str(h_conv))
        # print('h_pool' + str(layer_index) + ' = ' + str(h_pool))

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        reduced = int(h_pool.shape[1]) # or [2]
        input_channel = output_channel
        output_channel = output_channel * 16
        W_fc1 = weight_variable([reduced * reduced * input_channel, output_channel])
        b_fc1 = bias_variable([output_channel])
        h_pool_flat = tf.reshape(h_pool, [-1, reduced * reduced * input_channel])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        # dropoutの設定(過剰適合を排除)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # print('W_fc1 = ' + str(W_fc1))
        # print('b_fc1 = ' + str(b_fc1))
        # print('h_pool_flat = ' + str(h_pool_flat))
        # print('h_fc1_drop = ' + str(h_fc1_drop))

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        input_channel = output_channel
        output_channel = num_classes
        W_fc2 = weight_variable([input_channel, output_channel])
        b_fc2 = bias_variable([output_channel])
        # print('W_fc2 = ' + str(W_fc2))
        # print('b_fc2 = ' + str(b_fc2))

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # print('y_conv = ' + str(y_conv))

    # 各ラベルの確率を返す
    return y_conv


def calculate_loss(logits, labels):
    """ lossを計算する関数
    @param
        logits          ロジットのtensor, float - [batch_size, num_classes]
        labels          ラベルのtensor, int32 - [batch_size, num_classes]
    @return
        cross_entropy   交差エントロピーのtensor, float
    """
    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels * tf.log(logits))

    tf.summary.scalar("cross_entropy", cross_entropy)

    return cross_entropy


def training(loss, learning_rate):
    """ 訓練のOpを定義する関数
    @param
        loss            損失のtensor, loss()の結果
        learning_rate   学習係数
    @return
        train_step      訓練のOp
    """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def calculate_accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数
    @param
        logits      inference()の結果
        labels      ラベルのtensor, int32 - [batch_size, num_classes]
    @return
        accuracy    正解率(float)
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar("train_accuracy", accuracy)

    return accuracy


def image_to_vector(path, image_size=28):
    """ 画像のベクトル化
    @param
        path                画像path
        image_size          画像の1辺のpixel数
    @return
        vectorized_image    vectorized image
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (image_size, image_size))
    # Unrolling Parameters
    img = img.flatten()
    # feature normalize
    vectorized_image = img.astype(np.float32) / 255.0
    return vectorized_image


def checkpoint_exists(path):
    """ checkpointファイル(.ckpt)の確認
    @param
        path                checkpoint保存path
    @return
        vectorized_image    vectorized image
    """
    return all([
        os.path.isfile('%smodel.ckpt.index' % (path)),
        os.path.isfile('%smodel.ckpt.meta' % (path)),
        os.path.isfile('%scheckpoint' % (path))
    ])


def run_inference_on_image(image_path, labels, image_size=28, patch_size=5, output_channel=16, add_channel=2, layer=2):
    """ 画像の分類
    @param
        image_path              画像path
        labels                  分類ラベル配列
        image_size              画像の1辺のpixel数
        patch_size              畳み込みの切り出しサイズ
        output_channel          最初の層のノード数
        add_channel             最初のノード数から以降乗算で増やす数
        layer                   層数
    @return
        results ラベルとスコアの配列
    """
    # 画像のベクトル化
    image_vector = image_to_vector(image_path, image_size=image_size)

    # ニュラールネットワークモデルを生成
    feature_size = image_size * image_size * 3 # 特徴数 = ピクセル数 * rgb
    images_placeholder = tf.placeholder("float", shape=(None, feature_size))
    keep_prob = tf.placeholder("float")
    logits = inference(
        images_placeholder,
        keep_prob,
        num_classes=len(labels),
        image_size=image_size,
        patch_size=patch_size,
        output_channel=output_channel,
        add_channel=add_channel,
        layer=layer
    )

    # load session
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH + 'model.ckpt')

    # classify
    scores = logits.eval(feed_dict={images_placeholder: [image_vector], keep_prob: 1.0 })[0]
    results = list(map(lambda result:
        {
            'label': result[0],
            'score': str(result[1])
        }
    , zip(labels, scores)))
    results = sorted(results, key=lambda result: result['score'], reverse=True)

    # reset session
    tf.reset_default_graph()

    return results


def run_training(datasets, num_classes=10, image_size=28, max_steps=10, batch_size=100, learning_rate=1e-4, patch_size=5, output_channel=16, add_channel=2, layer=2):
    """ トレーニングの実行
    @param
        datasets                データセットタプル(train_images, test_images, train_labels, test_labels)
        num_classes             分類数
        image_size              画像の1辺のpixel数
        max_steps               トレーニング実行回数
        batch_size              1回のトレーニングに使用する画像枚数
        learning_rate           学習率
        patch_size              畳み込みの切り出しサイズ
        first_output_channel    最初の層のノード数
    """
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    # datasets.csvからデータセットを取得
    # - vectrizeされた画像(特徴)
    # - one of k方式のラベル(答え)
    train_images, test_images, train_labels, test_labels = datasets

    # トレーニング開始
    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        feature_size = image_size * image_size * 3 # 特徴数 = ピクセル数 * rgb
        images_placeholder = tf.placeholder("float", shape=(None, feature_size))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, num_classes))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = inference(
            images_placeholder,
            keep_prob,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            output_channel=output_channel,
            add_channel=add_channel,
            layer=layer
        )
        # loss()を呼び出して損失を計算
        loss_value = calculate_loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, learning_rate)
        # 精度の計算
        accuracy = calculate_accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()

        # セッション開始
        sess = tf.Session()

        # 前回の結果があれば上書き、なければ初期化
        if checkpoint_exists(CHECKPOINT_PATH):
            # checkpointの読み込み
            saver.restore(sess, CHECKPOINT_PATH + 'model.ckpt')
        else:
            # 変数の初期化
            sess.run(tf.global_variables_initializer())

        # tensorboard 可視化
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, graph=sess.graph)

        # 訓練の実行
        for step in range(max_steps):
            for i in range(int(len(train_images) / batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = batch_size * i
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: train_images[batch:batch+batch_size],
                  labels_placeholder: train_labels[batch:batch+batch_size],
                  keep_prob: 0.5})

            # 1 step終わるたびに精度を計算する
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels,
                keep_prob: 1.0})
            test_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: test_images,
                labels_placeholder: test_labels,
                keep_prob: 1.0})
            print('step {}, training accuracy {}, test accuracy {}'.format(
                step,
                train_accuracy,
                test_accuracy
            ))

            # 1 step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

        # 最終的なモデルを保存
        save_path = saver.save(sess, CHECKPOINT_PATH + 'model.ckpt')

        summary_writer.close()





