# tensorflow image custom classify

## what is this?

画像をtensorflowを用いて独自に定義した分類を判定する
データセットは/workspace/datasets/imagenet/readme.mdをみてください

- 0:飛行機
- 1:車
- 2:鳥
- 3:猫
- 4:鹿
- 5:犬
- 6:カエル
- 7:馬
- 8:船
- 9:トラック

## versions

- python      Python 2.7.13 :: Anaconda 4.3.0 (x86_64)
- tensorflow  v1.0.0
- opencv3     3.1.0

## TODO

32x32の画像を28x28に変換して学習している。
cifar-10の画像は精度が高いが、他から落としてきた長方形高画質画像の場合、ズレが生じて威力を発揮しない？？

モデルを引き継いで学習は可能

途中から分類数は増やせない(方法はあるっぽい)
分類数を最初から1000とかかなり大きく作るとその分精度は落ちる

## doc

http://kivantium.hateblo.jp/entry/2015/11/18/233834

## installation
install
- tensorflow
- opencv

## datasets

/workspace/datasets/imagenet/readme.md

## flow

1. /workspace/datasets/imagenet/readme.mdを読みつつ10クラスに分類された画像をトレーニング用とテスト用を生成する

```
$ python ../datasets/imagenet/convert_cifar10.py --file ../datasets/imagenet/cifar-10-batches-bin/data_batch_1.bin --format jpeg --out ./images/train --length 1000
$ python ../datasets/imagenet/convert_cifar10.py --file ../datasets/imagenet/cifar-10-batches-bin/test_batch.bin --format jpeg --out ./images/test --length 100
```

2. create_list_text.pyを実行してtrain.txtとtest.txtを生成

```
$ python create_list_text.py
```

3. training.pyでトレーニングの実行

```
$ python training.py --max_steps 10 --batch_size 10
or
$ python training.py
step 0, training accuracy 0.104
step 1, training accuracy 0.161
step 2, training accuracy 0.223
step 3, training accuracy 0.249
step 4, training accuracy 0.284
step 5, training accuracy 0.306
step 6, training accuracy 0.333
step 7, training accuracy 0.34
step 8, training accuracy 0.334
step 9, training accuracy 0.375
step 10, training accuracy 0.381
...
test accuracy 0.88
```

4. run.pyで試す

```
$ python run.py images/test/9/11.jpeg
$ python run.py images/test/9/11.jpeg images/test/0/03.jpeg
```

## tensorboard

- tensorboardの書き出し方はv1.0から変更されているので注意

|v0.12|v1.0|
|:---:|:---:|
|tf.scalar_summary()|tf.summary.scalar()|
|tf.merge_all_summaries()|tf.summary.merge_all()|
|tf.train.SummaryWriter()|tf.summary.FileWriter()|

```
$ tensorboard --logdir=/tmp/data
or
$ python /anaconda/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=/tmp/data
```
