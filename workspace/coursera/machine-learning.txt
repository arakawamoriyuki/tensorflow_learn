----------------------------------------
# coursera machine-learning
[coursera machine-learning](https://www.coursera.org/learn/machine-learning)
----------
## week1


### 機械学習の定義
t
  タスク、クラス(分類)値、2値なら0or1
p
  確率、パフォーマンス計測値、トレーニング用のデータが得られる確率
e
  経験


### 教師あり学習 supervised

回帰問題
  面積から住宅の価格予測
  連続値出力の予測 = 実数(数値)を求める
2値分類
  悪性良性腫瘍
  離散値出力の予測
  真偽値(や分類)を求める

n値分類に対応したアルゴリズム、サポートベクタマシンもある

### 教師なし学習 unsupervised

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


### 線形回帰

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


### 目的関数、コスト関数

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
----------





----------------------------------------
# TensorFlow Tutorialの数学的背景
[TensorFlow Tutorialの数学的背景](http://enakai00.hatenablog.com/entry/2016/02/29/121321)

----------
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










