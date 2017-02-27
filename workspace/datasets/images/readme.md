# datasets

## what is this?

cifar-10を利用した10クラスに分類されたデータセットを生成する

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

## documents

http://www.buildinsider.net/small/booktensorflow/0201

## download cifar-10-binary

```
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar zxf cifar-10-binary.tar.gz
```

## run

- トレーニング用10クラス計10000枚のjpeg書き出し
```
python convert_cifar10.py --file ./cifar-10-batches-bin/data_batch_1.bin --format jpeg --length 10000
```

- テスト用10クラス計1000枚のjpeg書き出し
```
python convert_cifar10.py --file ./cifar-10-batches-bin/test_batch.bin --format jpeg --length 1000
```
