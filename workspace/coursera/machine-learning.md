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

[lecture-slides3 pdf](https://d3c33hcgiwev3.cloudfront.net/_6e5172607f1af1b6156c070104ca213c_Lecture3.pdf?Expires=1487635200&Signature=gVv-R7jJMrJM-xBiOJ2PJZSg6goA7jgaNQIrpXxtVvYlYcd6AHLzYR4RQMdfbxFVI~2CURQwo2WnyLwlh98Tq-70Q9XOJH2B~8aiWIxvZkQtcQz3s2fB9rqlkQ7JKfzsI-Gn97aRm0MUTD88d4Ab-JmP9eHkjU6FbMRUGmmH0Mk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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

各プログラミング言語と同じで、start indexが0だったり1だったりする
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


#### メトリクスとベクトル同士の掛け算

    m*n matrics × n*1 matrics = m*1 matrics
    A * x = y
    yi = Ai * x

一次関数の素敵な計算

家のサイズ 4つのデータ
4×2 metricsで、

    1, 2104
    1, 1416
    1, 1534
    1, 852

予測の一次関数 hθ(x) = -40 + 0.25x
2×1 metricsで

    -40
    0.25

二つのベクトルを掛けると

    1×-40+2104×0.25
    1×-40+1416×0.25
    1×-40+1534×0.25
    1×-40+852×0.25

とか、または単に

    hθ(2104)
    hθ(1416)
    hθ(1534)
    hθ(852)

になる。
ソフトウェアで実装する時は1行で書ける！

    予測価格 = データ行列 × パラメータ
      = n×n metrics ×[θ0, θ1]

上記の例で言えば、返される予測価格は4次元メトリクス(ベクトル?)になる。

スカラ = 実数 (eg. 10.0)
ベクトル = 1次元配列 (eg. [1, 2, 3])
メトリクス = 2次元配列 (eg. [[1, 2], [3, 4], [5, 6]])

#### メトリクスとメトリクスの乗算

片方の辺の列を分割して列分メトリクスとベクトルの掛け算を行い、列個の結果を合わせる

    1, 3, 2
    4, 0, 1

×

    1, 3
    0, 1
    5, 2

=

    1, 3, 2
    4, 0, 1

×

    1
    0
    5

と

    1, 3, 2
    4, 0, 1

×

    3
    1
    2

=

    1 + 0 + 10
    4 + 0 + 5

と

    3 + 3 + 4
    12 + 0 + 2

= (合わせて)

    11, 10
    9, 14


になる

式としては

    m×n metrics × n×o metrics = m×o metrics

メトリクスとメトリクスの乗算を行うケースとして、
家の価格と、それを予測する仮説が3ケースあるとする。
以前までは1ケースだったのでメトリクス×ベクトルで計算可能だったが、
3ケースある場合にメトリクスでかけて求める事が出来る。

メトリクスとメトリクスの掛け算の場合、一般的な掛け算のように、右辺と左辺の順番を変えることはできない。

    A×B != B×A

ただし、

    (A×B)×CやA×(B×C)

は可能。結合則を満たすという。


#### 単位行列

ある数に1を掛けると下記の講式が成り立つ

    1×z = z×1 = z

メトリクスも同様に対角線上に並んだ1と他を0で埋めたメトリクスを掛けると同様。

    1, 0, 0
    0, 1, 0
    0, 0, 1

これを単位行列といい、Iで表す。
また明示的にIn×nと表す事もある。

前の項で

    A×B != B×A

だと説明したが、片方が単位行列だと上記はイコールになる。

    A×I == I×A


#### 線形代入、行列演算、インバース

数には逆数が存在する
逆数とはある数に何かを掛けると1になるような数字の事。
例えば

3の逆数 = 3(3**-1) = 1で、
**-1は単に⅓
⅓が3の逆数。

12の逆数は1/12

0には逆数がない。何を掛けても1にはなれないから。

インバース=行列の逆数？
インバースの求め方。

    A(A**-1) = A**-1 × A = I（単位行列）

m×m metricsは正方行列という。
メトリクスは、正方行列だけが逆行列を持つ

逆行列はライブラリで実装されていて簡単に計算可能。
octaveでは
qinv関数に行列を渡すとインバースが返される。

    A = 1, 2
        3, 4
    I = qinv(A)
    > 1, 0
    > 0, 1

0の正方行列はシンギュラー行列（特異行列）とか、縮退行列とよばれる。

まとめ

- 正方行列だけがインバースを持つ。持たない場合もある。
- 0の正方行列はインバースを持たない。


#### メトリクストランスポーズ（転置）

まずは例

A=

    1, 2, 0
    3, 5, 9

であれば、Aの転置は下記の記号で表され、

A**T=

    1,3
    2,5
    0,9

となる。
Pythonのzip的な動き。Rubyでいう配列のtransposeメソッド。

考え方としては、行列に左上から右下に線を描き、フリップしたようなイメージ！

講式は下記のようになり、__2×3 metrics__であれば__3×2 metrics__に次元がかわるのでインデックスが変わることに注目。

    Aij = Bji

#### まとめ

メトリクス + メトリクス

    [[1,2,1,2], [2,3,2,3]] + [[1,2,3,4], [1,2,3,4]] = [[2,4,4,6], [3,5,5,7]]

メトリクス * 数値

    [[1,2,3,4], [5,6,7,8]] * 2 = [[2,4,6,8], [10,12,14,16]]

メトリクス * ベクトル

    __m×n metrics × n×o metrics = m×o metrics__
    [[1, 2], [3, 4]] * [[5], [6]]
      = [[(1*5) + (2*6)], [(3*5) + (4*6)]]
      = [[5 + 12], [15 + 24]]
      = [[17], [39]]

メトリクス * メトリクス

    [[1, 2], [3, 4]] * [[5, 6], [7, 8]]
      = [[1, 2], [3, 4]] * [[5], [7]] & [[1, 2], [3, 4]] * [[6], [8]]
      = [[(1*5) + (2*7)], [(3*5) + (4*7)]] & [[(1*6) + (2*8)], [(3*6) + (4*8)]]
      = [[5 + 14], [15 + 28]] & [[6 + 16], [48 + 32]]
      = [[19], [43]] & [[22], [80]]
      = [[19, 22], [43, 80]]

メトリクス転置 トランスポーズ

    [[1,2], [3,4]] ** T = [[1,3], [2,4]]


## week2

複数の線形回帰と線形回帰のベストプラクティス
Octave and MATLAB

### Environment setup instructions

#### install

[GNU Octave 3.8.0](http://sourceforge.net/projects/octave/files/Octave%20MacOSX%20Binary/2013-12-30%20binary%20installer%20of%20Octave%203.8.0%20for%20OSX%2010.9.1%20%28beta%29/GNU_Octave_3.8.0-6.dmg/download)

[MATLAB](https://www.mathworks.com/licensecenter/classroom/machine_learning_od/)
* required mathworks login

```
$ vi ~/.zshrc

## octave
#
alias octave='/usr/local/octave/3.8.0/bin/octave-3.8.0; exit;'

$ octave
```

#### documents

[Octave documents](http://www.gnu.org/software/octave/doc/interpreter/)
[Octave(qiita)](http://qiita.com/tobira-code/items/7cc278da4e93555e9484)

[MATLAB documents](http://jp.mathworks.com/help/matlab/)

#### MATLAB 使い方

MATLABは統合開発環境？
octaveはirb的な？octaveもビジュアライズ可能

- 実行、保存
- 変数定義
  + 定義済み変数piなどもある
- 計算式の書き方
  + ほぼプログラミングと同じ
- 定義済み関数
  + expやsin,cos,tanなど
  + メトリクスを渡せたりする
- ベクトルとメトリクスの定義
  + ベクトルはカンマで、メトリクスはセミコロンで区切る
  + [[1,3], [2,4]]これが、[1,3;2,4]こうなる
  + インデックスは1始まり
  + 配列のアクセスはv(1)
  + v.^wはメトリクスの値を二乗する。v.* wは掛ける
  + v’はvのトランスポーズ。逆数。
  + Δ = デルタ = 差分 = d
  + グラフにおけるt1は最初の値。tnは最後の値。dtはプロット間隔
  + ベクトルの作成は2:0.2:3で2から0.2間隔で3までのベクトル作成
  + linspace関数でもベクトル作れる

```
a = [1,3;2,4]
b = 2:0.2:3
bt = b'
```

- ビジュアライズ
  + plot関数にxとyを渡す。それぞれ型はベクトル
  + 引数で色変えたり、タイトル関数とかレジェンド関数とかでグラフをカスタマイズ

    x = linspace(0, 2*pi, 100);
    y = sin(x);
    plot(x, y);

- メトリクスの作り方
  + セミコロンで区切るだけ
  + 対数とか0埋めとか関数が用意されてる
  + 配列へのアクセスで、M([1,2],3)とか範囲で取ったりできる。
  + 範囲は1:3などがシンタックスシュガー。単に:だけで全範囲。
  + lengthやsizeで個数や各行や列の個数のベクトルが取れる
  + [metrics、metrics]でメトリクスの結合が出来る
  + メトリクス同士の掛け算が出来る。

    octave> a = rand(2,2)
    a =
       0.23991   0.98035
       0.44349   0.70277
    octave> b = rand(2,2)
    b =
       0.26942   0.21830
       0.55178   0.78043
   octave> length(a)
   ans =  2
   octave> size(a)
   ans =
      2   2
    octave> c = [a,b]
    c =
       0.23991   0.98035   0.26942   0.21830
       0.44349   0.70277   0.55178   0.78043
   octave> a*b
   ans =
      0.60558   0.81746
      0.50726   0.64527

- プログラミング
  + コメントは#
  + or演算子や否定演算子など
  + メトリクスを数値と比較するとベクトルで0か1が帰る
  + if eles endで条件分岐
  + forでループ、whileもある

[Octave(qiita)](http://qiita.com/tobira-code/items/7cc278da4e93555e9484)を見るだけで問題ない


#### 複数の変数、複数のフューチャーの線形回帰

今までの家の広さだけでなく、ベッドルームの数や築年数も含めて学習する
この場合、特徴をx1,x2などで表す
nは特徴の数を表す
iは今まで通りインデックスで、xiはi番目のトレーニングセットである事を示す
jは特徴のインデックス

今までの特徴が一つの仮説

    hθ(x) = θ0 + θ1x

複数の場合の仮説の式は転置を使って省略できる

    hθ(x) = θ0 + θ1x1 × θ2x2 ×… θnxn
    = θTx

多変量の線形回帰という。


### 最急降下法 gradient descent

[lecture-slides4 pdf](https://d3c33hcgiwev3.cloudfront.net/_7532aa933df0e5055d163b77102ff2fb_Lecture4.pdf?Expires=1488153600&Signature=LuL0lvZ4hPYhZwyZ2J0F2BwlImG8QgTZ02fhiUCyJ4os4Tn9DjGeIS4iIIumzqXMRLWFJ8yVvy0JP~EHhMU~ilP3PVsYsDGxCtN5jdNIxAS~WClBwc1me7w43D9oxMz8rh4ghdG3QOLeIWmZ8DzX2Rw7eE2K7QKZRpNUnARGieY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

TODO: 動画内の問題が理解できてない
TODO: この節、動画は何度も見直した方が良い。完全に理解出来るまで。完全に理解できれば複数特徴の線形回帰の実装が出来る。

最急降下法では下記のようにパラメータを更新していく。

θ = パラメータのベクトル
a = 学習率

    repeat {
      θj := θj - a*σ/σθj*J(θ)
    }

```math
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}
```

偏微分の項であるσ/σθj*J(θ)を見ていく

θi0 = 1で定義していたので新しいアルゴリズムは古いアルゴリズムと似ている、同じという認識。

####フューチャースケーリング

複数の特徴を最急降下法で分析する場合、それぞれの値の単位を調整すると収束しやすくなる。
例えば、ctrやimp,clickなど。ctrは1以下の値でclickは大きい数、またimpはそれより100倍近く大きい。
だいたいこれぐらいという値で各特徴を割って同じ範囲にする手法。

だいたい-1以上1以下にする。
-3から3も経験上いい感じ。
-0.3から0.3も。

####ミーンノーマライゼーション 平均ノーマライゼーション

だいたい2000feetの家のサイズなら、
サイズに-1000して0へ平均化し、2000で割ってフューチャースケーリングする。

μiはここでいう0へ平均化するために引く1000を表し、siは値としての平均である2000を表す（最大引く最小）。
siは標準偏差を設定してもいい！

ある程度スケーリングしたら大幅に収束しやすくなるのである完全でなくていい。

#### 学習率 ラーニングレート

実際に学習させていくと、θは減少していく。
100回目と200回目のθを調べる。
増加していればそれは発散しているので小さい値にすべき。
上がり下がりする場合もある！その場合も小さくすべき。

0.001,0.003,0.01,0.03,0.1,0.3,1
と範囲分けして、線引きしていくのがいい感じ！

#### 適切なフィーチャーの選択

必ずしも複数の特徴を使う必要はない。
幅と奥行きなら、面積を使って1つの特徴とする事も出来る。
下記のような二次関数は急激に上がっても後に下降してしまう。

    hθ(x)=θ0+θ1(size)+θ2(size)

なのでルートを使う事で奥に行くほど平坦な線をかける。

    hθ(x)=θ0+θ1(size)+√θ2(size)

1から1000のサイズのデータがあるならば、
√1000=だいたい32なので

    x1=size/1000, x2=√size/32

でスケーリング出来る。
この手法を多項式回帰という。

#### 正規方程式 normal equation

最急降下法と同じレイヤのアルゴリズム。

仮説を立てて何度も計算する最急降下法とは違い、一発で手法を求める。


求め方は偏微分=0とする方法。
ここで、特徴のメトリクスXはX0に1がだ代入されている状態とする。

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10.06.16.png?expiry=1488153600000&hmac=0N2kJ-rPoRQ6XnCk_CiYwN5Fx-gH48ckAORayv2yCYk)

```
θ = (X' * X)^-1 X'y
```

`(X' * X)^-1`は逆行列。octaveで書くと

```
pinv(x' * x) * x' * y
```

pinv関数は逆行列を計算する。
正規方程式はフィーチャースケーリングをする必要はない。

##### 最急降下法と正規方程式のメリットデメリット

|最急降下法|正規方程式|
|:-:|:-:|
|学習率を選択しないといけない|学習率いらない、シンプル|
|多くの繰り返しが必要|繰り返しもいらない、一発|
|数百万の特徴があっても正しく動作する|特徴の数の3乗で計算処理が増える|
|感覚的に10000個以上の特徴で使う|感覚的に10000個以下の特徴で使う|

- 後に学習するロジスティック回帰や分類などのロジックでは正規方程式は実際にはうまくいかない場合がある。

#### 正規方程式と非可逆性

正規方程式を使う時に行列を逆行列にしてるけど、逆行列にできない特異行列だったらどうするの？
それは滅多に遭遇しない問題ではあるが。

octaveのpinv関数は正常に動作するように作られてる。(技術的な話になるので省略する)

inv = インバース(逆行列)にする関数
pinv = インバース(逆行列)にするが、目的のΘがちゃんと取れる関数

特徴を選択する上で、feet(フィート)とm(メートル)があれば、片方を削除してもいい。
同じような意味を持つ特徴は削除してもいい(すべき)。
正規化を用いて特徴を絞る。正規化についてはweek後半に習う。

多分、普通のメトリクスは下記が成り立つ。それを可逆性といい、それができない行列が非可逆性の行列という。

```
a == a''
a == inv(inv(a))
```

まぁどちらにせよ、octaveなどのライブラリがpinv関数などを用意しているのでそれを利用するので問題ない。


#### (宿題の提出方法)

- 問題の関数を作る
- .mで保存してoctave上で関数が動作するか確認する
- octave上でsubmitを実行
- 何を提出するか聞かれるので数字を選択して提出する
- メールアドレスとパスワードを入力すると送信され、すぐに答えが返る

[提出方法やデバッグに関するヒント](https://www.coursera.org/learn/machine-learning/supplement/SFKpu/programming-tips-from-mentors)

### octave 使い方

[lecture-slides5 pdf](https://d3c33hcgiwev3.cloudfront.net/_41759bf2241607b07a5d4cd1285bff6b_Lecture5.pdf?Expires=1487894400&Signature=DBB0liAzvnnbS7yJKl-jVq6tWeJDc1QwrNfVLLauiFJ0~dwiwETFM1O1g3SYFkIotLctRqBjGw8ptw-jvjD5kOTSWyb7G0dg3FboUnPDSsfHSoX4~PHlJE7g043feWndrpZwmCHkgrFTZmNsc0ZDI9RzNDnY9Gg~aufhMHmGphE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

MALTABと基本的に一緒！

- PS1コマンドで見やすく
- !=は~=
- hist関数でメトリクスをグラフ化
- eye関数で単位行列作成。多分identityの駄洒落
- help eyeでヘルプみれる
- データの保存場所はpwd、lsとかcd出来る
- loadコマンドでデータ読み込み saveで保存。mat拡張子でバイナリ、txtとアスキーオプションで人間が読める
- whoとwhosコマンドで現在の変数
- クリアで変数クリア

- A(:)でメトリクス全要素をベクトルで取得
- A()はメトリクスへのアクセスで、代入も可能
- ./など、ドットは各要素に対しての計算
- lengthとonesを利用して各要素インクリメント。単純に+1でもいいけど
- [v,i] = max(A)で、最大とそのインデックスをとれる

- printでpng出力できる
- figureで複数グラフ表示
- imagescでカラーマップ表示

```
v=1:10
for i=v,
  disp(i);
end;
while i <= 5;
  disp(i);
  i = i + 1;
end;
while true;
  disp(i);
  if i == 10;
    break;
  elseif i < 10;
    disp('i < 10');
  else;
    disp('else');
  end;
end;
```

- 関数は拡張子を.mにして読み込む

```
function y = myfunction(x)
y = x^2;
```

```
function [y1, y2] = myfunction(x)
y1 = x^2;
y2 = x^3;
```

- パスの追加可能

```
$ addpath('/Users/arakawa/Desktop')
```

- 関数の実行

```
$ [y1, y2] = myfunction(x)
```

#### octave ベクトル化

すべての式はベクトル化して計算できるのであれば
ループを使わずベクトル化して計算すべき

- 線形回帰の例 hΘ(x) = ΘTx

```
prediction = theta' * x
```


## week3

### TODO:

----------


----------------------------------------
# Tips

## Anaconda in Python

[参考](http://qiita.com/t2y/items/2a3eb58103e85d8064b6)

- 主要ライブラリをオールインワンでインストール
- condaというパッケージ管理システム入り(pipの代わり)
- condaでバンージョン管理も可(pyenvの代わり)
- condaで仮想環境管理も可(virtualenv/venの代わり)


1. [anaconda](https://www.continuum.io/downloads#osx)公式からGUI(もしくはCUI)でインストール

2. インストールした場所によるが(試した際はユーザーにインストール出来なかった)、インストールすると下記にnumpyなど機械学習に必要な環境が作られる。

```
/anaconda
```

```
$ /anaconda/bin/python --version
Python 2.7.13 :: Anaconda 4.3.0 (x86_64)
$ /anaconda/bin/pip list | grep numpy      # condaでパッケージ管理するのでpip入ってはいるけどconda通したほうが良い
numpy (1.11.3)
numpydoc (0.6.0)
$ /anaconda/bin/conda list | grep numpy
numpy                     1.11.3                   py27_0
numpydoc                  0.6.0                    py27_0
```

3. 現system環境と競合(pythonコマンドなど)するので使いたいときだけPATH通したほうが良い。

```
$ vi ~/.zshrc
```

```
## path functions
#
path_append ()  { path_remove $1; export PATH="$PATH:$1"; }
path_prepend () { path_remove $1; export PATH="$1:$PATH"; }
path_remove ()  { export PATH=`echo -n $PATH | awk -v RS=: -v ORS=: '$0 != "'$1'"' | sed 's/:$//'`; }

~~~

## anaconda
#
ANACONDA_PATH = /anaconda/bin
anaconda_active () {
  path_prepend $ANACONDA_PATH
}
anaconda_deactive () {
  path_remove $ANACONDA_PATH
}
```

```
$ python --version
Python 2.7.11
$ source ~/.zshrc
$ anaconda_active
$ python --version
Python 2.7.13 :: Anaconda 4.3.0 (x86_64)
$ anaconda_deactive
$ python --version
Python 2.7.11
```

## OpenCV cv2

- 画像や動画を加工するpythonのライブラリ
- 線画にしたり顔を検出したり？
- 機械学習では、CNN(畳み込みニューラルネットワーク)で、白黒や輪郭情報などの学習に必要な要素にフィルタしてデータを最適化する。

```
# anaconda環境前提
$ conda install -c https://conda.anaconda.org/menpo opencv3
$ python
>>> import cv2
>>> cv2.__version__
'3.1.0'
```

## TensorFlow

- 機械学習ライブラリ

[画像分類](https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet)
[画像分類 inception](https://github.com/tensorflow/models/tree/master/inception/inception)

[画像キャプション](https://github.com/tensorflow/models/tree/master/im2txt)

[自然言語CNN 畳み込みニューラルネットワーク](https://github.com/dennybritz/cnn-text-classification-tf)
[RNN](https://github.com/tensorflow/models/tree/master/tutorials/rnn)
[教師なし学習](https://github.com/tensorflow/models/tree/master/tutorials/embedding)

keras-rl
OpenAiGym
OpenAiUniverse

### install

詳しくは[ここ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#anaconda-installation)

```
# とりあえずsystem pipとかanaconda pipでinstallしてしまっていたら消しておく。
$ pip uninstall tensorflow
$ pip3 uninstall tensorflow

# anaconda環境でcondaで作成した環境に対してtensorflowをinstall
$ conda create -n tensorflow python=2.7
$ source activate tensorflow
(tensorflow)$ conda install -c conda-forge tensorflow
(tensorflow)$ python
>> import tensorflow as tf
>> tf.__version__
'1.0.0'

# anaconda環境でグローバル環境に対して?tensorflowをinstall
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl # githubを参照！！ Mac OS X, CPU only, Python 2.7
$ pip install --ignore-installed --upgrade $TF_BINARY_URL
>> import tensorflow as tf
>> tf.__version__
'1.0.0'
```

## Chainer

- 機械学習ライブラリ

### Models

[線画着色](https://hub.docker.com/r/liamjones/paintschainer-docker/)
[イラスト生成](http://qiita.com/mattya/items/e5bfe5e04b9d2f0bbd47)

## jupyter

- pythonで実行可能なコードとその結果を併せて保存するwebページを起動する
- anacondaはbundleされてるのでinstall必要なし

```
# install
$ pip install jupyter
# 起動
$ jupyter notebook
```

## matplotlib

- pythonでグラフを表示する
- anacondaはbundleされてるのでinstall必要なし

```
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-3, 3, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show() # jupyter notebookで実行する場合は必要なし
```

## OpenAiGym

- DQN(強化学習)評価用のプラットフォーム

```
$ git clone http://github.com/openai/gym
$ cd gym
$ pip install -e .
```

## OpenAiUniverse

```
$ git clone http://github.com/openai/universe
$ cd universe
$ pip install -e .
```

* エラーが出る場合はエラー内容に従って下記をインストール
* 公式のチュートリアルコードで、import universeする前にimport zbarlightする!(zbarlight必須)

- appstoreからxcode update
- docker
- numpy (anacondaならbundle済み)
- incremental (anacondaならbundle済み?)
- golang
- go-vncdriver
- libjpeg-turbo
- zbar(brew)
- zbarlight(pip)

```
$ xcode-select --install
$ pip install numpy incremental
$ brew install libjpeg-turbo golang
```

### docker

- 最初だけの忘れがちコマンド

```
$ docker-machine create --driver virtualbox default
$ docker-machine start default; eval "$(docker-machine env default)"
```

### golang

```
$ brew install libjpeg-turbo golang
$ vi ~/.zshrc
## golang
#
export PATH=$PATH:/usr/local/opt/go/libexec/bin
```

### go-vncdriver

```
$ git clone https://github.com/openai/go-vncdriver.git
$ cd go-vncdriver
$ python build.py
$ pip install -e .
```

- 設定 zshrcに書いとく必要があるっぽい

```
## go-vncdriver
#
go_vncdriver_active () {
  export GOPATH='/Users/arakawa/Documents/repository/tensorflow_learn/go-vncdriver/.build'
  export CGO_CFLAGS='-I/Users/arakawa/anaconda/lib/python2.7/site-packages/numpy/core/include -I/Users/arakawa/anaconda/include/python2.7'
  export CGO_LDFLAGS='/usr/local/opt/jpeg-turbo/lib/libjpeg.dylib -undefined dynamic_lookup'
  export GO15VENDOREXPERIMENT='1'
}
```

### zbar

```
$ brew install zbar
$ pip install zbarlight
```

## keras-rl

- tensorflowのDQN(強化学習?)深層学習ラッパー

[github](https://github.com/matthiasplappert/keras-rl)

```
$ pip install keras-rl h5py
$ git clone https://github.com/matthiasplappert/keras-rl.git
$ cd keras-rl
$ python examples/dqn_cartpole.py
```

----------------------------------------
# TensorFlowで学ぶディープラーニング入門

## TensorFlow入門

### 考え方

- 与えられたデータを元にして道のデータを予測する数式を考える

    y = w0 + w1x + w2x2 + w3x3 + w4x4

- 数式に含まれるパラメータの良し悪しを判断する誤差関数を用意する

    E = 1/2*Σ12,n=1(yn - tn)**2

- 誤差関数を最小にするようにパラメータの値を決定する(tensorflowの役目)

### ニューラルネットワーク

複数の仮説関数(ノード)にパラメータを通して得られた結果をパラメータとして、
さらに仮説関数に通していく事で結果をより正しく構成していく。
1層のノード数を増やしたり、層自体を増やしたりする。

### ディープラーニング 深層学習

単にニューラルネットワークでノードや層を増やすだけでは処理能力に限界がある。

CNN(畳み込みニューラルネットワーク)では、1層目のノードで畳み込みフィルタという関数でデータを最適化する。
畳み込みフィルタは、画像で言えば、白黒や輪郭情報など、学習に必要な要素のみにフィルタしてデータを最適化する事で分析しやすくしている。

畳み込み層の次のプーリング層では、画像の解像度を落とすなど。

リカレントニューラルネットワーク(RNN)では、自然言語処理の最適化が行われる。具体的には
this is a penという言葉を解析して、thisの次にくる言葉の確率を計算し、言語が正しいかを評価する。
連続した言語は中間層に保存され、長文に対応する。中間層の値を次の入力に再利用するニューラルネットワークがRNN。


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
