----------------------------------------
# coursera machine-learning
[coursera machine-learning](https://www.coursera.org/learn/machine-learning)
----------
## week1

### 1 Introduction

[lecture-slides1 pdf](https://d3c33hcgiwev3.cloudfront.net/_974fa7509d583eabb592839f9716fe25_Lecture1.pdf?Expires=1487548800&Signature=ePccut9XD5ZVv0GzM13mY2vCzfddpTptfLEbifH~5icLwwUA7k7K97jd7WK5~J9JcT3Mg37I9oQ3gDqGOtS4cwzD2lmJDorAbjOpiAhxNP11-MiHlp-SAxa7KtctdlOdcH-J14xkPLyIECUxp9PMtKzJZl05rQnwFD6IOgKrxzg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

#### 機械学習の定義
t
  タスク、クラス(分類)値、2値なら0or1
p
  確率、パフォーマンス計測値、トレーニング用のデータが得られる確率
e
  経験


#### 教師あり学習 supervised

回帰問題
  面積から住宅の価格予測
  連続値出力の予測 = 実数(数値)を求める
2値分類
  悪性良性腫瘍
  離散値出力の予測
  真偽値(や分類)を求める

n値分類に対応したアルゴリズム、サポートベクタマシンもある

#### 教師なし学習 unsupervised

答えが割り振られてないデータセットで学習

クラスタリングアルゴリズム
  グーグルニュースの記事振り分けやDNAから人の特徴を判別するなど
  答えがないデータセットの集合を分類する
  分類を作成する学習、元から何種類あるのかすら分からないデータセットを分類する
  自然言語のマインドマップとかデータの関連性を調べる感じ？

カクテルパーティアルゴリズム
  多人数の会話の音を複数箇所から録音し、クラスタリングアルゴリズムで分類して人毎の音に分ける
  複雑なコードで実装しそうだが、クラスタリングアルゴリズムで1行で書ける

octaveは学習、試作に良い


### 2 Model and Cost Function

[lecture-slides2 pdf](https://d3c33hcgiwev3.cloudfront.net/_ec21cea314b2ac7d9e627706501b5baa_Lecture2.pdf?Expires=1487548800&Signature=CJ1SDVqhdnvZQO3RErtrq5TJer2B1OKsAQ8Tgr9Me30dIte15btdbZNU~eQ9zV-qP8IPR6FMqMERCiLgcYv58onN0qKGaRGDKu19er8~u4~D3lScBENggWXgsTqlTHp6nhDfZSJPxDElw~POpXLHbpZkYnL9S4~sYmEaZ3F-s0k_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

#### 線形回帰

m = 訓練サンプル数
x = 入力変数、特徴
y = 出力変数、目標変数
(x, y) = 一件の訓練サンプル
(x(i), y(i)) = 複雑の訓練サンプル(カッコは乗ではなくインデックス)
i = インデックス、現在の訓練行

h = 適用したいアルゴリズムの関数(仮説という意味だけど意味としては間違いかも、古い慣例)

家の価格をわりだすのなら、サイズであるxをアルゴリズムに渡して結果のyを取る
y = h(x)

単回帰
入力値xが一つの線形回帰
初歩の初歩


#### 目的関数、コスト関数

theta(シータ) = θiはモデルのパラメータ
θ0は開始地点
θ1は傾き

h(x) = θ0 + θ1*x

プロットされた点からθ0とθ1を求めれば求めたい線を書ける

h(x)とyを最小化したい
1/2m(h(xi) - yi)**2 = コスト関数

(h(xi) - yi)**2 = 入力値と実際の価格を引いた総和の二乗を求める
1/2m？？ = 2はいくらか計算を楽にするため。求めるための振り幅？

標準偏差に似た計算？
minimizeなどを含めた数式を目的関数
二乗誤差関数と呼ばれたりする

j = 目的関数 = θを求める

```math
J(θ0,θ1)=12m∑i=1m(y^i−yi)2=12m∑i=1m(hθ(xi)−yi)2
```

```math
\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2
```


重要要素
- 仮説関数 hθ(x)
  家の価格を返す
- 目的関数 J(θ1)
  直線の傾きを返す

(1,1)(2,2)(3,3)のグラフがあれば

J(0)
= 1/2m × (1**2+2**2+3**2)
= ⅙ × 14
= 14/6

目的関数 J(θ1)の値が最小になるパラメータが直線となる？

```

# j = コスト関数、θ1を求める為のアルゴリズム

# 1/2m( (h(xi) - yi)**2 )
# m = 3, 1/6( (x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2 )

# θ0 = 開始地点
# θ1 = 傾き
# m = データ数
# y = 実データの値(eg. 家の値段)
# x = 入力(eg. 家の広さ)

# θ1を調査する一次関数、実データ
@line = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
# @line = [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
# @line = [[1.0, 0.2], [2.0, 0.4], [3.0, 0.6]]
# @line = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] # @range = 30
# @line = [[1.0, 1.0], [2.0, 0.5], [3.0, 0.0]] # 傾きの調整が必要

# 実データ個数
@m = @line.size

# コスト関数を使って調査する範囲、傾きに応じて変更する必要あり。大きいほど正確
@range = 20

def j theta_1

  line = @line.dup
  line.unshift(result = 0.0)

  # right = (h(xi) - yi)**2
  right = line.reduce do |result, point|
    x = point[0]
    y = point[1]

    # (h(xi) - yi)**2
    result += ((theta_1 * x) - y) ** 2.0
  end

  # 1/2m
  left = 1.0 / (2.0 * @m)

  return left * right
end

# θ1を仮説で値を入れ、最小の値が正解
min_θ1 = nil
min_j = nil
@range.times do |index|
  θ1 = (index * 0.1).round(2)
  puts "θ1 = #{θ1}, j(θ1) = #{j(θ1)}"

  # 最小値を保存
  min_j = j(θ1) if min_j.nil?
  min_θ1 = θ1 if min_θ1.nil?

  min_θ1, min_j = θ1, j(θ1) if min_j > j(θ1)
end

# 最小である1.0がもっともらしい。θ1 = 1.0
puts "minimize θ1 = #{min_θ1}"
```

定数まとめ
- 仮説
  h(x) = θ0 + θ1*x
- パラメータ
  θ0, θ1
- 目的関数
  J(θ0, θ1) = 1/2m( (h(xi) - yi)**2 )
- 最適化の目的
  minmize J(θ0, θ1)
   θ0, θ1

前回のJ(θ0)は二次元プロットできたが、
今回(今後)のJ(θ0, θ1)は三次元プロット、または等高線図で図を表現する


#### 最急降下法

gradient descent method?

minmize J(θ0, θ1)

JでΘを最小化したい場合
θ0 = 0など仮説を立てて、その値を少しずつ変化させて最小値を探す。
しかし、その最小値は局所的最小値かもしれない、底が複数ある皿のイメージ。卵パックとか？

数学的にアルゴリズムを見る。

    repeat until convergence {
      Θj := Θj - (α * (∂ / ∂Θj)) J(θ0, θ1)   (for j = 0 and j = 1)
    }

|記号|名前|説明|
|:-:|:-:|:-:|:-:|
|:=|代入演算子|数学では__a := b__、プログラミングで言う__a = b__|
|=|単に証明、表明|数学では__a = b__、プログラミングで言う__a == b__|
|α|アルファ|学習率、大きいほど下る一歩が大きくなる|
|∂|デル、(パーシャル)ディー|偏微分|

(α * (∂ / ∂Θj))は導関数項という。後ほど説明する。
ポイントは(for j = 0 and j = 1)で繰り返すので、
Θj = Θ0, Θ1になり、Θ0とΘ1を更新する点。
更新する際は__Θ0 = Θ0 - 何か__と__Θ1 = Θ1 - 何か__になる

もう一つポイント。実装する上で、同時に更新できる

    temp0 = Θ0 - 何か
    temp1 = Θ1 - 何か
    Θ0 = temp0
    Θ1 = temp1

という実装にすること。

    temp0 = Θ0 - 何か
    Θ0 = temp0
    temp1 = Θ1 - 何か
    Θ1 = temp1

だとΘ0に代入後にΘ1を(Θ0に依存した式で)求めることになってしまう。

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1487548800000&hmac=d7L9YI5t8Zo49Z9BIIn02U6JJhdzz-un6V-JT-ZRDws)

#### 導関数項

わかりやすくするため、導関数項をh(θ1)だけで考える
θ1 = θ1 - a(d/θ1×d), J(θ1)
微分とはこの場合、J(θ1)のボウル型曲線グラフの現在地に接続する一次関数の線を引き、その傾きが何かということ
その線が正の傾きであればa(d/θ1×d)は何か正の数になり、左に移動する。
負の傾きなら負の数になり、右に移動する。
そうして最小値に近づける

学習率aについて考える
学習率が小さいと時間がかかるが、大きすぎると逆に最小値から遠ざかる恐れがある

最小値になると導関数項、微分は0となり、その場にとどまる
学習率の調整は重要

#### 線形回帰

最急降下法と二乗誤差のコスト関数を使って作る。

線形回帰の3dグラフは必ずボウル型になる。
今まで見た山と丘がある3dにはならない。
凸型関数という。局所的最適解はない。必ず一つの底に行き着く。
最急降下法はバッチアルゴリズムと呼ばれる場合もある。

ポイント
- 局所的最適解はなく、一つの底だけ。ただし、目的関数Jが変わると底が複数になり、その限りではない。
- aは固定でも底に収束する。ただし、大きすぎると無理。


```math
\begin{align*} \text{repeat until convergence: } \lbrace & \newline \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline \rbrace& \end{align*}
```

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/QFpooaaaEea7TQ6MHcgMPA_cc3c276df7991b1072b2afb142a78da1_Screenshot-2016-11-09-08.30.54.png?expiry=1487548800000&hmac=Q7Yegaz3dmYHZvOARbW5dHxYz3VsT8Cz-oX6GY3Pa70)


#### 数式まとめ
h = 仮説関数
h(xi) = hΘ(xi) = θ0 + (θ1 * xi)
j = 目的関数 = 最小を求める
j(Θ0, Θ1) = 1/2m((hΘ(xi) - yi)**2)


### 3 Linear Algebra Review

#### ベクトル

Linear Algebra = 線形代数
行列 = ベクトル = メトリクス = Metrics
大括弧の中に数字をたくさん書く数学記法
二次元配列
R2*4 = 数学記法

    [[1,2,3,4], [5,6,7,8]] = 2x4 metrics = R2*4 = 行x列 metrics

    [1,2,3,4] = 2x4 metrics
     5,6,7,8

参照方法
Aij = A配列のi index内のj index
プログラミングでいうと

    A = [[1,2,3,4], [5,6,7,8]]
    A[i][j]
    A[2][3] == 7
    A[3][5] == undefined //プログラミングと同じ

ベクトルとはnx1 metricsの事
データの数が決まっていない状態

    [1,2,3,4] = 2x4 metrics
    行の数から上記は4次元ベクトル
    または単にR4と表記する

    yi = i th element
    y1 = 1
    y3 = 3

各プログラミング言語と同じで、start indexが0だったり1だったりする
数学的には1indexが一般、機械学習の実装時には0が主流
これよりのちは1indexで説明している

行列を参照する時は数学的には一般的に大文字を使う(A,B,X,Y)
生の数値などは小文字を使う(a,b,x,y)

#### スカラー乗算

ベクトル同士の演算に使う。

ベクトル足し算
ベクトル同士の足し算は、配列の要素同士の足し算。
もちろん同じmetrics同士の足し算しかできない

    [[1,2,1,2], [2,3,2,3]] + [[1,2,3,4], [1,2,3,4]] = [[2,4,4,6], [3,5,5,7]]

スカラ乗算
ベクトル同士の乗算も単純に
配列の各要素に対して乗算を行う

    [[1,2,3,4], [5,6,7,8]] * 2 = [[2,4,6,8], [10,12,14,16]]

また、それらを連続して計算する場合もある

    3 * A + B - C / 3 = (3*A) + B - (C/3) =


TODO: ベクトル同士の乗算
----------


----------------------------------------
# TensorFlowで学ぶディープラーニング入門
TODO

----------------------------------------
# ゼロから作るDeepLearning - Pythonで学ぶディープラーニングの理論と実装
TODO

----------------------------------------
# TensorFlow Tutorialの数学的背景
[TensorFlow Tutorialの数学的背景](http://enakai00.hatenablog.com/entry/2016/02/29/121321)

## MNIST For ML Beginners（その1）

### 平面データの分類問題
2次元平面に◯と×の2種類
f(x0,x1) = 1次関数 = 線
σ(f(x0,x1)) = シグモイド関数 = Easing.easeInOutQuad
f(x0,x1)の値を「データが◯である確率 y」
![図](http://cdn-ak.f.st-hatena.com/images/fotolife/e/enakai00/20160214/20160214153322.png)

    # トレーニングデータを入れるn*2配列、x = 軸、[x[0], x[1]]
    x = tf.placeholder(tf.float32, [None, 2])
    # 係数(2x+1なら2)、分類を入れる、w = 分類、[w[0], w[1]]
    w = tf.Variable(tf.zeros([2, 1]))
    # 定数項(xやyなど未知数を含まない値、2x+1なら1)、
    b = tf.Variable(tf.zeros([1]))
    # シグモイド関数の定義、 y は、「placeholderである x に M 個のデータを入れると、Mx1 行列の値が得られる関数」
    y = tf.sigmoid(tf.matmul(x, w) + b*mult)




----------------------------------------
# Mathematics Cheet Sheet

### 微分

ある地点での変化量
直線では変化量は変わらないがy=x**2の湾曲した一次関数では座標によって変化量は変わる
時速40kmは1時間に40km進むことだが、それは平均の速度であり、微分である瞬間の速度を求めると0kmだったり60kmだったりする


### 積分

xを時間、yを速度としたグラフを考える。
この場合、グラフをエリアチャートにした面積が積分となる。


###1次関数

- f(x0,x1)
- Easing.linear
- 線

```math
f(x_0, x_1)
```


###シグモイド関数

- σ(f(x0,x1))
- Easing.easeInOutQuad
- なめらかな線

```math
y_i = \sigma(f(x_{0i},x_{1i}))
```


###尤度関数(ゆうどかんすう)

TODO:

```math
{f(x)={1 \over \sqrt{2\pi\sigma^{2}}} \exp \left(-{1 \over 2}{(x-\mu)^2 \over \sigma^2} \right)}
```
![図](https://qiita-image-store.s3.amazonaws.com/0/50670/c0ae2048-1432-f6c8-e871-49da757055bf.png)


###最尤推定値(さいゆうすいていち)

TODO:

![図](http://mathtrain.jp/wp-content/uploads/2015/07/mle.png)


###目的関数、コスト関数、二乗誤差関数

1/2m(h(xi) - yi)**2

```math
J(θ0,θ1)=12m∑i=1m(y^i−yi)2=12m∑i=1m(hθ(xi)−yi)2
```

```math
\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2
```


###確率密度関数

TODO:


###正規分布 == ガウス分布

![図](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Standard_deviation_diagram.svg/600px-Standard_deviation_diagram.svg.png)

平均値の付近に集積するようなデータの分布を表した連続的な変数に関する確率分布


###標準偏差

データのばらつきの大きさの指標

標準偏差とは「データのばらつきの大きさ」を表わす指標で、各データの値と平均の差の2乗の合計をデータの個数で割った値の正の平方根として求められる

[参考](http://atarimae.biz/archives/5379)

平均点が60点のテストで70点取った

ごく一部の生徒が平均を下げただけで、普通に勉強したら80点以上取れるテスト
a_test =「0点、5点、10点、70点、80点、80点、82点、85点、93点、95点」(平均点60点)

多くの生徒が間違えた超難問のうちの1つを正解した
b_test =「50点、52点、54点、60点、60点、60点、61点、61点、70点、72点」(平均点60点)

「1番目の例の標準偏差は約36.67点→ばらつきの大きなテストだった→平均＋10点はスゴくない」
「2番目の例の標準偏差は約6.68点→ばらつきの小さいテストだった→平均＋10点はスゴイ」

偏差 = 各データの値と平均値の差

{(0－60)^2+(5－60)^2+(10－60)^2+(70－60)^2+(80－60)^2+(80－60)^2+(82－60)^2+(85－60)^2+(93－60)^2+(95－60)^2}÷10=1344.8

分散 = 1344.8^2 = データのばらつきの大きさ
標準偏差 = √1344.8 = 約36.67点

```
require 'complex'
avg = 60
a_test = [0,5,10,70,80,80,82,85,93,95]
b_test = [50,52,54,60,60,60,61,61,70,72]
[{a: a_test}, {b: b_test}].each do |type, test|
  test.unshift(result = 0)
  # 分散値の計算
  dispersion = test.reduce {|result, next|
    result + ((next - avg) ** 2)
  }
  # 標準偏差の計算
  deviation = Math.sqrt(dispersion)
  puts "#{type} #{deviation}"
end
```


# 勾配降下法(こうばいこうかほう)

= 最急降下法?
= gradient descent method?

J(θ0, θ1)を三次元プロットすると
ボウル型の3Dグラフができる。
それを辿って底につくイメージ。
等高線図での円の中心を割り出す。

----------------------------------------
# TensorFlow Cheet Sheet

## methods

### トレーニングデータ

    x = tf.placeholder(tf.float32, [None, 2])

### 変数

    w = tf.Variable(tf.zeros([2, 1])) #係数定義
    b = tf.Variable(tf.zeros([1])) #定数項定義

### 行列の掛け算

    tf.matmul(x, w)

```math
f(x_0, x_1)
```

### シグモイド関数

    tf.sigmoid(tf.matmul(x, w) + b*mult)

```math
y_i = \sigma(f(x_{0i},x_{1i}))
```

### 行列の全ての要素を足し合わせる関数

    tf.reduce_sum(y_*tf.log(y) + (1-y_)*tf.log(1-y))
