# tensorflow image classify

## what is this?

1400万枚以上の画像を2万カテゴリに分類されているimagenetの
データを用いて学習したGoogle製モデルinception-v3を利用して
画像の内容とスコアを返す

## versions

- python      Python 2.7.13 :: Anaconda 4.3.0 (x86_64)
- tensorflow  v1.0.0

## documents

- https://www.tensorflow.org/get_started/os_setup
- https://www.tensorflow.org/tutorials/image_recognition/

## installation

- install tensorflow

```
cd {this project root}
git clone git@github.com:tensorflow/models.git
python tensorflow_learn/models/tutorials/image/imagenet/classify_image.py
```

## shell command

```
python classify_image.py --image_file images/cat_toast.jpg
python classify_image.py --image_file images/tractor.jpg
```

## shell function

```
classify_image () {
    python /path/to/models/tutorials/image/imagenet/classify_image.py --image_file $1
}
```

## shell script

classify image with ./images directory in .jpeg files

```
sh classify_image.sh
```

### result
```
mousetrap (score = 0.30130)
Egyptian cat (score = 0.15552)
tabby, tabby cat (score = 0.03588)
schipperke (score = 0.03222)
spindle (score = 0.02587)

comic book (score = 0.42162)
sandal (score = 0.08734)
book jacket, dust cover, dust jacket, dust wrapper (score = 0.06787)
cuirass (score = 0.02211)
stage (score = 0.01681)

pill bottle (score = 0.12647)
carton (score = 0.10339)
lotion (score = 0.08676)
menu (score = 0.05403)
sunscreen, sunblock, sun blocker (score = 0.02374)

comic book (score = 0.37137)
umbrella (score = 0.06459)
maillot (score = 0.06339)
carton (score = 0.05708)
bikini, two-piece (score = 0.03452)

brassiere, bra, bandeau (score = 0.18686)
gown (score = 0.11268)
bikini, two-piece (score = 0.09364)
maillot (score = 0.05720)
web site, website, internet site, site (score = 0.03620)

picket fence, paling (score = 0.09681)
confectionery, confectionary, candy store (score = 0.06130)
web site, website, internet site, site (score = 0.03501)
barbershop (score = 0.03331)
envelope (score = 0.03253)

tractor (score = 0.68600)
harvester, reaper (score = 0.13954)
forklift (score = 0.03088)
plow, plough (score = 0.01352)
snowplow, snowplough (score = 0.00969)
```

### result translated

```
マウストラップ（スコア= 0.30130）
エジプト猫（スコア= 0.15552）
tabby、tabby cat（スコア= 0.03588）
シッパル（スコア= 0.03222）
スピンドル（スコア= 0.02587）

漫画本（スコア= 0.42162）
サンダル（スコア= 0.08734）
ブックジャケット、ダストカバー、ダストジャケット、ダストラッパー（スコア= 0.06787）
cuirass（スコア= 0.02211）
ステージ（スコア= 0.01681）

丸薬（スコア= 0.12647）
カートン（スコア= 0.10339）
ローション（スコア= 0.08676）
メニュー（スコア= 0.05403）
サンスクリーン、サンブロック、サンブロッカー（スコア= 0.02374）

漫画本（スコア= 0.37137）
傘（スコア= 0.06459）
maillot（スコア= 0.06339）
カートン（スコア= 0.05708）
ビキニ、ツーピース（スコア= 0.03452）

ブラジャー、ブラ、バンドウー（スコア= 0.18686）
ガウン（スコア= 0.11268）
ビキニ、ツーピース（スコア= 0.09364）
マイヨール（スコア= 0.05720）
ウェブサイト、ウェブサイト、インターネットサイト、サイト（スコア= 0.03620）

ピケフェンス、ペリング（スコア= 0.09681）
菓子、菓子、キャンディー・ストア（スコア= 0.06130）
ウェブサイト、ウェブサイト、インターネットサイト、サイト（スコア= 0.03501）
理髪店（スコア= 0.03331）
包絡線（スコア= 0.03253）

トラクター（スコア= 0.68600）
ハーベスター、リーパー（スコア= 0.13954）
フォークリフト（スコア= 0.03088）
プラウ、プラウ（スコア= 0.01352）
除雪、除雪（スコア= 0.00969）
```