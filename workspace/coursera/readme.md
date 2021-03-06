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

####フィーチャースケーリング

複数の特徴を最急降下法で分析する場合、それぞれの値の単位を調整すると収束しやすくなる。
例えば、ctrやimp,clickなど。ctrは1以下の値でclickは大きい数、またimpはそれより100倍近く大きい。
だいたいこれぐらいという値で各特徴を割って同じ範囲にする手法。

だいたい-1以上1以下にする。
-3から3も経験上いい感じ。
-0.3から0.3も。

####ミーンノーマライゼーション 平均正則化

だいたい2000feetの家のサイズなら、
サイズに-2000して0へ平均正則化し、-1000で割ってフューチャースケーリングする。

μiはここでいう0へ平均化するために引く1000を表し、siは値としての平均である2000を表す（最大引く最小）。
siは標準偏差を設定してもいい！

ある程度スケーリングしたら大幅に収束しやすくなるのである完全でなくていい。


例.

|中期試験点数(x)|中期試験点数の二乗(x^2)|最終試験点数(y)|
|:-:|:-:|:-:|
|89|7921|96|
|72|5184|74|
|94|8836|87|
|69|4761|78|

中期試験点数の二乗をフューチャースケーリングと平均正則化を適用する

```
x_vector = [7921, 5184, 8836, 4761]

max = 8836.0
min = 4761.0
avg = x_vector.reduce(:+) / x_vector.count # = 6675.0
normalize = max - min # = 4075.0

x_vector.each do |x|
  puts "x(#{x}) => #{(x - avg) / normalize}"
  x
end
```

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

#### インターセプト項 定数項

data(1, :)に1を代入してベクトル計算する

```
data = % m*n metrics
x = data(:,1); % m vector
X = [ones(m, 1), data(:,1)];
```

#### クイズ

1. フィーチャースケーリング計算

2. 次にとる行動として正解は？
15回 線形回帰を実行
a=0.3
J(Θ)を計算
J(Θ)は急速に減少し次にレベルオフ?
レベルオフ = 平坦になったって事っぽい。

3. 下記の次元数は何？
n=3 #特徴数
m=14 #教師データ個数

4. 最急降下法と正規方程式どっち？
- m=1000000
- n=200000
- 多変量線形回帰

5. フィーチャースケーリングする理由


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

#### octave演習

##### グラフ表示エラー

```
$ octave
> figure;
gnuplot> set terminal aqua enhanced title "Figure 1" size 560 420  font "*,6.66667" dashlength 1
                      ^
         line 0: unknown or ambiguous terminal type; type just 'set terminal' for a list

WARNING: Plotting with an 'unknown' terminal.
No output will be generated. Please select a terminal with 'set terminal'.

$ brew cask install xquartz
$ brew cask install aquaterm
$ brew uninstall gnuplot
$ brew install gnuplot --with-aquaterm --with-x11
```

##### 単特徴の線形回帰

```
% load data
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);

% sample data
% X = rand(m, 1);
% y = rand(m, 1);
% m = 10;

% plot data
plotData(X, y);

% intercept term (X1には1のインターセプト項を挿入する)
X = [ones(m, 1), data(:,1)];

% theta initialize
theta = zeros(2, 1);

% ループ回数
iterations = 1500;
% 学習率
alpha = 0.01;

% コストの計算
% # J = (1 / (2 * m)) * ((h(xi) - yi)^2)
function J = computeCost(X, y, theta)
  m = length(y);
  costs = ((X * theta) - y) .^ 2;
  J = sum(costs) / (2 * m);
end

% 最急降下法
% # repeat until convergence {
% #   Θj := Θj - (α * (∂ / ∂Θj)) J(θ0, θ1)   (for j = 0 and j = 1)
% # }
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
    h = X * theta;
    errors = h - y;
    delta = X' * errors;
    theta = theta - (alpha / m) * delta;
    J_history(iter) = computeCost(X, y, theta);
  end
end

theta = gradientDescent(X, y, theta, alpha, iterations);

plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')

predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...predict2*10000);

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
  t = [theta0_vals(i); theta1_vals(j)];
  J_vals(i,j) = computeCost(X, y, t);
  end
end

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
```

##### 多特徴の線形回帰

```
% --- 多特徴の線形回帰 ---

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% 標準偏差
function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  numColumns = size(X, 2);
  mu = mean(X);
  sigma = std(X);
  for i = 1:numColumns
      X_norm(:,i) = (X(:, i) - mu(i)) / sigma(i);
  end;
end

% 標準偏差の計算
[X mu sigma] = featureNormalize(X);
% intercept term (X1には1のインターセプト項を挿入する)
X = [ones(m, 1) X];

% 多特徴のコストの計算
function J = computeCostMulti(X, y, theta)
  m = length(y);
  cost = 0;
  for i = 1:m
    cost = cost + (theta' * X(i,:)' - y(i))^2;
  end;
  J = cost / (2 * m);
end

% 多特徴の最急降下法
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
    h = X * theta;
    errors = h - y;
    delta = X' * errors;
    theta = theta - (alpha / m) * delta;
    J_history(iter) = computeCostMulti(X, y, theta);
  end
end

% 学習率
alpha = 0.01;
% ループ回数
num_iters = 400;

% theta initialize
theta = zeros(3, 1);

% 多特徴の最急降下
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

x = [1 1650 3]';
price = theta' * x;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...'(using gradient descent):\n $%f\n'], price);
fprintf('Program paused. Press enter to continue.\n');



% --- 正規方程式 ---

% オレゴン州の[家の広さ,部屋数,価格]
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% intercept term (X1には1のインターセプト項を挿入する)
X = [ones(m, 1) X];

% 正規方程式
function [theta] = normalEqn(X, y)
  theta = pinv(X' * X) * X' * y;
end

% 正規方程式の計算
theta = normalEqn(X, y);
size = 1650;
rooms = 3;
x = [1 size rooms]';
price = theta' * x;
```


## week3

[lecture-slides6 pdf](https://d3c33hcgiwev3.cloudfront.net/_964b8d77dc0ee6fd42ac7d8a70c4ffa1_Lecture6.pdf?Expires=1488931200&Signature=PQwvN14QhjjyT2pvzSb8cHYZob6iyu3iL9xMzpbG31xVCiwJpfI0Pwaxtv1JoVeUzN2L83iVQc8g75QDTbiJvsU7hjBnJZ4OUc-5trNMzI3D387Pw5C1Gg5JXYs6ByLO~y-kEtqUGBa5xEpGCvX-7fS4xmXQQFgBO88xAeXtOIU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

[lecture-slides7 pdf](https://d3c33hcgiwev3.cloudfront.net/_7d030d67103ce0e7f39dee1d7f78525c_Lecture7.pdf?Expires=1489363200&Signature=MOqd8cMOlT9c-Td4fh-lS8dMETNeXHHBv1sLdvq6ZJgT94L5NQ3q4Rz0f2YUOifVovKou1~6IE7tAfqpC5HwmTRve0djLk-Npo9VTcwBJG5D0twDJPG4Cex9EPIXdP1WErIEW4tqvGoGPo68Rw5k2fZTTwybLJpOz0BuE3M8K80_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

### 分類

ロジスティック回帰

メールがスパムか、盗まれたクレジットカードとかパスワードか、悪性腫瘍か良性腫瘍か
まずは2つの分類問題（バイナリ分類問題）

    y∈ {0, 1}

    if hθ(x) ≧ 0.5 then y = 1
    if hθ(x) > 0.5 then y = 0

直線だとデータセットの特徴の大きさによって傾きがだいぶ変わるため、目的関数に線形回帰を用いるべきではない

ロジスティック回帰では必ず0か1を返すように実装する

計算式

    hθ(x) = g(θtx)
    g(z) = 1/1+(e**-1)

Octaveなら

    h = g(θ’ * x)

g(z)は、zは実数でシグモイド関数に渡すことを意味する
また、ロジスティック関数とシグモイド関数は同じ意味でどちらもgを意味する

シグモイド関数を通して式にすると

    hθ(x) = 1/1+(e**-(θ’ * x))

ここで、xはいつものようにインターセプト項を付け加えた特徴とのベクトル

    x = [
    1
    size (腫瘍サイズ)
    ]

hθ(x) = 0.7なら1と判断

もう一つ、1である確率が7割の場合の式

    hθ(x) = P(y =1|x;0) = 0.7

P(y =1|x;0)は1である確率を示し、P(y =0|x;0)は0である確率を示すので下記の公式が成り立つ

    P(y =1|x;0) + P(y =0|x;0) = 1
    P(y =0|x;0) = 1 - P(y =1|x;0)

### 決定境界 (Decision boundary)

前回のビデオでは下記の公式が成り立つ事を説明した

    if hθ(x) ≧ 0.5 then y = 1
    if hθ(x) > 0.5 then y = 0

下記のパラメータを割り振った場合、

    Θ0 = -3
    Θ1 = 1
    Θ2 = 1

下記の公式に当てはめた時

    h = g(θ’ * x)
    hΘ(x) = g(Θ0 + Θ1x1 + Θ2x2)

実際に値を展開すると下記になり、

    y = 1 if -3 + x1 + x2 ≧ 0
    y = 0 if -3 + x1 + x2 < 0

右辺に-3を移動してこう解釈もできる

    y = 1 if x1 + x2 ≧ 3
    y = 0 if x1 + x2 < 3


グラフ上に横軸をx1、縦軸をx2として書くと境界線は x1 + x2 = 3となり、
点(x1=0,x2=3)と点(x1=3,x2=0)を通る線になる

```octave
% 右上がy=1、左下がy=0
plot(linspace(0, 3), linspace(3, 0))
```

でy = 1とy = 0の境界線の事を決定境界という。
下記のパラメータが割り振られている場合の計算過程は下記。

    Θ0 = 5
    Θ1 = -1
    Θ2 = 0

    hΘ(x) = g(5 + (-1*x1) + (0*x2))
    hΘ(x) = g(5 - x1)

    y = 1 if 5 - x1 ≧ 0
    y = 0 if 5 - x1 < 0

    y = 1 if x1 ≧ 5
    y = 0 if x1 < 5

x1が5以上ならy=1で、x1が5以下ならy=0な線を作る

### 線ではない決定境界

決定境界が固まってなく、散らばっている場合の決定境界をどう作るか。
具体的には目的関数が代わり、θが5つになったりする。

```
hΘ(x) = g(θ0 + θ1x1 + θ2x2 + (θ3x1)^2 + (θ4x1)^2)
```

パラメータが下記のように割り振られている場合の計算過程は

```
Θ = [-1,0,0,1,1]

hΘ(x) = g(-1 + 0*x1 + 0*x2 + (1*x1)^2 + (1*x1)^2)

y = 1 if -1 + 0*x1 + 0*x2 + (1*x1)^2 + (1*x1)^2 ≧ 0
y = 1 if -1 + x1^2 + x1^2 ≧ 0
y = 1 if x1^2 + x1^2 ≧ 1
```

上の公式はイメージとして、原点(x1=0, x2=0)から半径1に決定境界が存在する
つまり、円状の決定境界が出来上がる例。
円の外側はy=1となり、内側はy=0となる

注意点として、あくまで決定境界の形はデータセットが持っている訳ではなく、
選択する目的関数によって形が変わってくる。
目的関数の選択方法は次回以降のビデオで学ぶが、それは理解してくべき。

もっと高次元な目的関数もある。
これらはもっと複雑な楕円などの形をしている。(複数の楕円がある場合もあるはず？)

```
hΘ(x) = g(θ0 + θ1x1 + θ2x2 + (θ3x1)^2 + (θ4x1)^2) + ((θ5x1)^2)*(x2^2) + (θ6x1)^3*x2 + ...
```


### 目的関数

例で理解する

```
# トレーニングセット
{(x1, y1)...(xm, ym)}
# サンプル数
m
# 特徴のベクトル
x = [x1...xm]
# x0は必ず1、インターセプト項 定数項として
x0 = 1
# yは0か1
y = {0 or 1}

# 分類における仮説、コスト関数
hθ(x) = 1 / ( 1+(e^-(θ' * x)) )
```

- 非凸関数とは(non-convex)

今までのコスト関数のプロットはボウル状で局所的最適解(局所的最小値)はなかった。
必ず斜面を下り、グローバルな最小値にたどり着くようになっている。
そういうグラフを凸関数という。

非凸関数は局所的にボウル状の場所がいくつかあり、最急降下法を適用しても
最も急な場所で下っていく方法なのでグローバルな最小にたどり着く事はできない。


分類における仮説、コスト関数は下記のようになっていて(乗があって複雑で)、
hθ(x)が非線形になる。JのΘは非凸の関数になる。

```
hθ(x) = 1 / ( 1+(e^-(θ' * x)) )
```

ただし、両辺の乗などを相殺する事で線形の式にする事も可能。
そういった最適化を行ったあとのロジスティック回帰で行うコスト関数は下記のようになる。

```math
\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}
```

```
Cost(hθ(x),y)=0 if hθ(x)=y
Cost(hθ(x),y)→∞ if y=0andhθ(x)→1
Cost(hθ(x),y)→∞ if y=1andhθ(x)→0
```

```
Cost(hθ(x),y)={
  -log(hθ(x)) if y=1
  -log(1 - hθ(x)) if y=0
}
```

![y=1図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1488672000000&hmac=kpO6JWndQ89kEn_oXaYdUNyEPtUnz6Eq54lC_rxPkNM)

![y=0図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1488672000000&hmac=9kTQkFn_ElpK_Ws7YOL4J70p1hMXzMvNNcWQmbMKI5U)

hθ(x)(横軸)が1の場合Cost(縦軸)は0で、
log関数は、hθ(x)が1の場合は0だが、hθ(x)が0に近づくに連れて∞に向かうような線を描く。

hθ(x) = 予測値
y = 答え
Cost = 推定が間違っていた場合のペナルティ

例えば

hθ(x) = 1, y = 1
の場合Cost = 0になる。
つまり、予測は1で答えも1だった場合にその予測へのペナルティはない事になる。

一方で、
hθ(x) = 0, y = 1
の場合、Cost = ∞になる。
実際に∞のペナルティを与える事はないが、その予測が0である確信が高いにも関わらず間違えた場合に予測へのペナルティは無限大に大きくなる。(∞は特異点なので、hθ(x)=0とかCost=∞になる事はないはず)

実際には
hθ(x) = 0.9, y = 1
の場合、ほぼ1だと予測して正解したので、Costは低い実数。('ほぼ'と断定できなかった事へのペナルティ)
hθ(x) = 0.1, y = 1
の場合、ほぼ0だと予測したが間違えたのでCostはかなり高い実数。(完全に間違えた事へのペナルティ)


- 断定して間違うほどコスト(推定が間違っていた場合のペナルティ)はでかくなる。
- 疑問として、hθ(x) = 0.5の場合、ペナルティがあんまりない事に違和感(毎回0.5と予測すれば学習できていると勘違いしない？最急降下法をなんどかループさせればhθ(x) = 0.5となる事も減るのか？)


### シンプルなコスト関数と勾配降下法

いままでは下記のようにy=1の場合は、y=0の場合はと分けていたが

```
Cost(hθ(x),y)={
  -log(hθ(x)) if y=1
  -log(1 - hθ(x)) if y=0
}
```

それらをシンプルにしたのが下記の式になる

```
Cost(hΘ(x), y) = -y*log(hΘ(x)) - (1-y)*log(1-hΘ(x))
Cost(hθ(x),y) = −ylog(hθ(x))−(1−y)log(1−hθ(x))
```

最終的にベクトルを含めたコストの計算は下記の式になる。(最尤法推計、覚えなくていい)

```
J(Θ) = 1/m*sum(Cost(hΘ(x), y))
```

そして最急降下法でΘを求める式は下記。
以前習ったロジスティック回帰と同じ。変わったのは目的関数の定義が変わった。

```
repeat {
  Θj := Θj - (a*sum(hΘ(xi) - yi)xij)
}
```

forでもいいが、ベクトル化した計算式はこうなる。

```
θ := θ - a/m * X' * (g(XΘ) - y)
```

### さらに早いロジスティック回帰のアルゴリズム(advanced optimization)

下記のアルゴリズムは特徴の数が多くなると勾配降下法に変わる早いアルゴリズム。
ただし、下記の説明はコースの範囲を超える。(何週間も勉強するハメになる)
ただ、こういうアルゴリズムがあるという事は覚えておいた法がいい。

- 共益勾配法(conjugate gradient)
- BFGS
- L-BFGS

上記アルゴリズムの特徴として

- 共通して学習率aを選ぶ必要がない(ラインサーチアルゴリズムという方法でいいレートの学習率を選択してくれる)
- 最急降下法より早い
- デメリットとしては、複雑。難しいアルゴリズムで、10年以上使ってきたが、中身まで知ったのはここ2~3年。つまり、中身を知らずとも使うことはできる。自前で実装すべきではない。ライブラリは各言語にあるが、中身がおかしいのもある。使うのであれば多数のライブラリを使って比べるといい。

octaveでは下記のように使用できる。

```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end

options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

### one vs all分類(one vs all classification)

複数クラス分類方法

天気やメールのフォルダわけなど
one vs all分類、one vs rest分類で分類する事ができる。

1つ対その他に分けて2クラス分類をする。
3つの分類の場合、3つの分類機ができる。

その３つの分類機の中からもっとも確度の高い分類が答えとなる。

```
max(hΘ[i](x))
```

### オーバーフィッティング問題(the problem of overfitting)

オーバーフィッティングとは

線形回帰の住宅価格の予想の例で
大きくなるにつれて平行に近づくため、一次関数は直線を描き、あまりフィットしない。
これをアンダーフィッティング、またはこのアルゴリズムは高バイアスだ、という。
二次関数ならフィットする。
5次関数にもなると、線はなだらかではなく、データセットのポイントはとおるが、ガタガタした線になる。

この5次関数でガタガタになり、良い予想にならないことをオーバーフィッティングといい、このアルゴリズムは高バリアンスだ、という。

    高バイアス <=> 高バリアンス
    アンダーフィッティング <=> オーバーフィッティング

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30.png?expiry=1489104000000&hmac=1V0y4oxpdKsaueeKvE3oLVxIDGr2eUNw1bjnKM_fuio)

オーバーフィッティングは特徴が多すぎる（θが多すぎる）と起こりうる問題

同じ問題はロジスティック回帰にもおこる。
分類問題で一次関数の直線ではもちろん良いモデルではなく、
二次関数ではある程度の間違いはあるもののほぼいい感じの決定境界の曲線ができる。
多次関数では、すごく頑張りすぎて全く間違いを含まない範囲の決定境界を取ろうとする（アメーバや滴った水の跡のような決定境界になる）

####オーバーフィッティングした時のデバッグ方法

オーバーフィッティングは特徴の次元が多い割に、データセットが少ない場合になりえる。

- 人力で特徴を減らす

人力で必要ない次元を減らすのは有効な手段。だが、情報を捨てることになる事も念頭に。
コースの後半にどの特徴が有効でどの特徴が有益でないか自動で判別する方法を学ぶ。

- 正規化を行う

特徴に重み付けを行い、その特徴の影響力を調整する。

### 正規化

正規化する為のコスト関数

二次関数のコスト関数にペナルティーを与えた特徴（二乗などしてかなり0にちかい）を足す。
具体的にはθ3やθ4に対して二乗するなど。

```
θ0+θ1x+θ2x2+θ3x3+θ4x4
minθ 12m [∑mi=1(hθ(x(i))−y(i))2+λ ∑nj=1θ2j]
```

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/j0X9h6tUEeawbAp5ByfpEg_ea3e85af4056c56fa704547770da65a6_Screenshot-2016-11-15-08.53.32.png?expiry=1489104000000&hmac=CRq0Ql6ZxpDuFkLQYdGZ7DjuorKrjN_W7w-V98leHNM)

λ（ラムダ）は正規化パラメータと呼ばれる
この正規化パラメータは大きいほどペナルティーを課し、より0に近づき、θ0だけの式に近づく事で直線化し、アンダーフィッティングする。
正規化パラメータが小さいほどグネグネとまがり、オーバーフィッティングする。

この正規化パラメータ（ラムダ）は後半で自動的に選択できるようになる。


### 線形回帰の正規化

線形回帰の正規化は普段の線形回帰のθ1からθnまでに
λ/m＊θjを足す。
θ0は特別で、足さない。

```
repeat {
  Θj := Θj - (a*sum( ((hΘ(xi) - yi) * xij) + (λ/m*θj))
}
```

```
repeat {
  Θj := Θj(1 - (a*(λ/m))) - (a*(λ/m))*sum( (hΘ(xi) - yi) * xij))
}
```

ここで、下記は0.99などの1よりほんの少し小さい値になる。
```
1 - (a*(λ/m))
```

### 正規方程式の正規化

```
θ = ((X'X + λ*L)**-1) * X'y

while L =[0
            1
              1
                ...
                  1]
```

ここでのLはm+1の逆行列メトリクスで、要素1-1が0のメトリクスになる。

[参照](https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression)


#### 非可逆性について(advancedな内容)

m(トレーニングセットの数) <= n(特徴の数)

の場合

X' (Xのインバース)

はシンギュラー行列（特異行列）とか、縮退行列と呼ばれる行列になる可能性(かなり低い)がある。

octaveのpinv関数は縮退行列に関してよきに計らって値を返してくれる。
octaveのinv関数はネイティブなインバース。
他の言語(pythonやtensorflow)でやる時にはその辺注意する。


### ロジスティック回帰の正規化

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Od9mobDaEeaCrQqTpeD5ng_4f5e9c71d1aa285c1152ed4262f019c1_Screenshot-2016-11-22-09.31.21.png?expiry=1489363200000&hmac=sGod8A43T7YYXHiMgnnukOYRnB3fJiOduzr_m4U8A_E)

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dfHLC70SEea4MxKdJPaTxA_306de28804a7467f7d84da0fe3ee9c7b_Screen-Shot-2016-12-07-at-10.49.02-PM.png?expiry=1489363200000&hmac=SJaGMZox2Zyd2xY5maHB0RWBeEZVhWbkpNBcT5GbYCw)


## week4

[lecture-slides8 pdf](https://d3c33hcgiwev3.cloudfront.net/_48018e8190fedff87b572550690056d2_Lecture8.pdf?Expires=1490227200&Signature=YvA0-ZjN8vDnfz9OdOKZHtr95dRsa2dh2zhqk9jB4wODRpl~yNClANC7u~1Cqa6MiCiU63U~QASJvNj6~z5RzhKKmY-mpPjTo9FyhczjyDK5yWES0vgo7uJdS-atrOteCsByle1Qjm1qWP-wNv~1wGI1-7ErZCLJqqwW9xF5n~o_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

### Motivations ニューラルネットワークに対するモチベーション

これまではフィーチャーの少ない分類や回帰を行なってきたが、フィーチャーが多くなるにつれて多項式がθの二乗割る2で複雑になっていく。
複雑で、オーバーフィッティングしやすく、計算コストも高くなる。

ほとんどの場合、機械学習のフィーチャーは多い。
車の画像を分類する例を挙げると、
白黒画像の50x50pixelでは0から250の値のフィーチャーが2500個出来ることになる。
カラーではx3になるので7500個になる。
2500個の場合、xi*xj個のパターンの多項式が出来るため、最終的に30000個になる

#### ニューラルネットワークの歴史

- ニューラルネットワークは脳を模したアルゴリズムで、1980年代からあり、最近までマシンスペックの問題などで下火になっていたが、スペックが揃った今日再注目されているアルゴリズムである。
- 脳について。聴覚皮質は声を理解する場所だが、神経接続という方法で目からの信号を送る事で見ることを学んだ。体性感覚皮質は触覚を処理するが、同様に見る事を学んだ。
- fdaが行なった実験で、舌にカメラからのグレースケール信号を電極を通して刺激する事で、盲目の人が10分そこらで物体を検知できるようになる。同様に、盲目の人が舌打ちや指鳴らしで跳ね返る音で物体を検知する人力ソナーもそう。カエルに第3の目を移植して、その目の使い方を覚えたりする。
- ニューラルネットワークは、この脳を模倣して、あらゆるロジックを実装せずに、学習させるアルゴリズム。

### Neural Networks ニューラルネットワーク

人間の脳のニューロンについての説明。

ニューロンを模したネットワークがニューラルネットワーク。
信号を複数受け取り、何らかの計算をして他のニューロンに信号を送る。
ニューラルネットワークの図は通常、x1からxnの信号を書くが、x0を書くときもある。x0はバイアスニューロンと呼ばれ、普通は必ず1。
アクティベーション関数とは、g(z)と一緒。シグモイド関数。
今まではθと書いてきたが、ニューラルネットワークの文献によってはweight(重み)という場合がある。

- バイアスニューロン = x0 = 1
- アクティベーション関数 = g(z)
- weight = θ =　パラメータ

ニューラルネットワークは、ニューロンの集まりを意味した図で、1層目は入力レイヤー、2層目はニューロンが受け取る中間層や隠れ層、最後の層でhθ(x)を出力する出力レイヤーと呼ばれる。

- 入力レイヤー = フィーチャーを渡す層
- 中間層 = 隠れ層 = フィーチャーを受け取り、次の層へ渡す。各ノードに別々のパラメータが割り振られる。
- 出力レイヤー = 最後の層、hθ(x)を出力する

aij = iはノード(ユニットの)インデックス、jは層

各ノードがパラメータの一部分を計算し(層の全てのノードにフィーチャーを割り振って計算し)、次の層に渡している。出力レイヤーではメトリクスがかえされるが、ノードの多さによってその幅と高さはかわる。

例えば、層が3つで、入力層に2つのノード、中間層に4つのノードがある場合

    sj = 2, s(j+1) = 4

と表して、下記の公式により、4 x 3のメトリクスが返される。

```
s(j+1) × (sj +1)
```

sjは入力層の個数で、s(j+1)は次の層。多分s(j+n)はn+1層目。


#### ニューラルネットワークの実装

gはシグモイド関数で、aは各層のノード配列、<>（本来は上付き丸括弧）の添字は層のインデックスとして、下記のような式とする。

```
a1<2> = g(z1<2>)
```

各々ののノードでの計算をベクトルで計算可能。

```
x0 = 1
x = [x0 x1 x2 x3]
z<2> = [z1<2> z2<2> z3<2>]
```

```
z<2> = θ<1>x
a<2> = g(z<2>)
```

このgは、各ベクトルの要素にシグモイド関数を適用する意味。a<2>もz<2>も（例ではどちらも3次元ベクトル）同じ次元のベクトル。

a<1>は入力レイヤーで、xをアクティベーションするので下記のように考えられるので、代入するとこうなる。

```
a<1> = x
z<2> = θ<1>a<1>
```

前説明したとおり、ニューラルネットワークの図には書かないが、バイアスユニットも存在する。それは常に1。なので、aは4次元ということになる。

```
add a0<2> = 1
a = 4x1 metrics (vector)
```

出力レイヤーの値のはこうなり、

```
z<3> = θ<2>a<2>
```

仮説の出力はこう表すことができ、フォワードプロバケーションともよばれる。

```
hθ(x) = a<3> = g(z<3>)
```

ニューラルネットワークの各ノードの計算は今までやったロジスティック回帰とにている。実際、層の添字や、ノードのインデックスがあるかだけで、それを抜けばロジスティック回帰と変わらない。

あんまりピンと来てないかもしれないが、次の2つのビデオでより鮮明になるはず。

[一連の計算式](https://www.coursera.org/learn/machine-learning/supplement/YlEVx/model-representation-ii)


### ニューラルネットワークの具体例

論理積andとその真理値表を使ったニューラルネットワークの例

バイアスのパラメータが-30
x1が20
x2が20
の場合、
-30 + (x1 × 20) + (x2 × 20)
になるので、両方1でなければ0以上を返せず、論理積になる。

バイアスのパラメータが-10
x1が20
x2が20
の場合、
-10 + (x1 × 20) + (x2 × 20)
になるので、両方1もしくは片方1の場合に0以上を返すので論理和となる。

![OR回路](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/f_ueJLGnEea3qApInhZCFg_a5ff8edc62c9a09900eae075e8502e34_Screenshot-2016-11-23-10.03.48.png?expiry=1490227200000&hmac=Addo3fvQpW1Icn5JzkSON5o-dUxitYiFBV3JknAsXIo)

バイアスのパラメータが10
x1が-20
の場合
否定(not)になる。

バイアスの重みが10で
x1が-20
x2が-20
の場合、両方0の場合に1を返す否定積？(not and not)になる。


これら3つの回路を利用して、排他的論理和を作る。
線では決定境界を描けない非線形なデータでも排他的論理和を使えば可能になる。

論理積と否定積から得た値を論理和に通すことで排他的論理和が可能。

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/rag_zbGqEeaSmhJaoV5QvA_52c04a987dcb692da8979a2198f3d8d7_Screenshot-2016-11-23-10.28.41.png?expiry=1490227200000&hmac=QwQnJtUXHYB8S3lA8EdFMY90xLgvVbYGJhW459vYazI)

### ニューラルネットワークを用いた分類

以前教えた1 vs allの応用になる。
人、車、バイク、トラックを分類する場合、出力レイヤーに4つのレイヤーを持つようにする。

y= 4vector = [1 0 0 0] = 人！

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/9Aeo6bGtEea4MxKdJPaTxA_4febc7ec9ac9dd0e4309bd1778171d36_Screenshot-2016-11-23-10.49.05.png?expiry=1490227200000&hmac=XYr2ka_Dq-4lW72ES0er5oAYlpZosDAGTLAOnyWn5uA)


## week5

[lecture-slides9 pdf](https://d3c33hcgiwev3.cloudfront.net/_1afdf5a2e2e24350ec9bad90aefd19fe_Lecture9.pdf?Expires=1490486400&Signature=F0Ub0Dgk2ohOyfruipYanrXw~AZOssM~eGJmvwNj7SyyYfODGFBugTYaf2zwWgqrYJmJJ4k3tksySHqNlcXUblcCQsi5aMH8wCNyqQKHmSDG~GSFZhZ9V4uduxj6iFuZ-yw5P6avb6ZZe9bpPCpgrauzTTZFVOkvdqAvg0l~N0w_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

### ニューラルネットワーク コスト関数

L = ニューラルネットワークのレイヤー数
l = レイヤーインデックス
s = ユニット(ノード)数
sl = lで指定されたレイヤーのユニット数(バイアスユニットは含めない)
K = sL = 出力レイヤーのユニット数

- バイナリ分類(0or1)
出力レイヤーのユニット数は1、sL = K = 1

- マルチクラス分類(K classes)
出力レイヤーのユニット数は3以上、K = sL >= 3

#### ニューラルネットワークのコスト関数

```
hθ(x) = k vector
(hθ(x))i = i th output
j(θ) = ...
```

```
J(Θ)=−1m∑i=1m∑k=1K[y(i)klog((hΘ(x(i)))k)+(1−y(i)k)log(1−(hΘ(x(i)))k)]+λ2m∑l=1L−1∑i=1sl∑j=1sl+1(Θ(l)j,i)2
```

ベクトル化の実装
```
Δ(L):=Δ(L)+δ(L+1)∗(a(L))T
```

[式の参考](https://www.coursera.org/learn/machine-learning/supplement/afqGa/cost-function)

### バックプロパゲーションアルゴリズム

コスト関数を最小化するアルゴリズム

トレーニングデータが1個しかない例で説明、単にxとyだけ。

1. フォワードプロバケーションで前方に計算した値を渡していく。

![フォワードプロバケーション](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bYLgwteoEeaX9Qr89uJd1A_73f280ff78695f84ae512f19acfa29a3_Screenshot-2017-01-10-18.16.50.png?expiry=1490400000000&hmac=lbjXqEqUeuOIcK6p8RPsnAv5QjPmIsPsn0n49n5H1cY)

2. バックプロパゲーションでデルタ項を計算する。

lは上付き添字として、jは下付き添字として、δj<l>と表す。それは、レイヤーlのj番目のユニット。デルタは誤差を表す。
アルファは入力、デルタは誤差。

[バックプロパゲーション](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ul6i5teoEea1UArqXEX_3g_a36fb24a11c744d7552f0fecf2fdd752_Screenshot-2017-01-10-17.13.27.png?expiry=1490400000000&hmac=axo4-vrhOUVGU4LBIDOLWGUG25gDY0UZHRniShrInGk)

```
δ = α - y

δ(誤差) = α(コスト関数が出した値) - y(答え)
α = h(x)なので、コスト関数が出したコスト関数値がどれだけ間違っているかの値がデルタ。
```

```
g′(z(l))=a(l) .∗ (1−a(l))
```

△ = Δ = 大文字のデルタ

[式の参考](https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm)

### バックプロパゲーションの直感 Backpropagation Intuition

バックプロパゲーションが何をしているか深くまで知る必要はない。
実装方法さえわかればブラックボックスでもよく働く。

もう少し直感的に、実装よりにフォワードプロパゲーションとバックプロパゲーションを紐解く。

#### フォワードプロパゲーション

下記のニューラルネットワークがある

1層目(入力層)ノード2個
2層目(中間層)ノード2個
3層目(中間層)ノード2個
4層目(出力層)ノード1個

- 3層目の2個目のノードの計算例

```
x1<3> = (Θ10<2> * 1) + (Θ11<2> * a1<2>) + (Θ12<2> * a1<2>)

3層目の2個目のノード =
  (theta[1][0] * バイアス) +
  (theta[1][1] * 3層目の1個目のノード) +
  (theta[1][2] * 3層目の2個目のノード)
```

バックプロパゲーションも似たような仕組み、計算方法

#### バックプロパゲーション

```
cost(t)=y(t) log(hΘ(x(t)))+(1−y(t)) log(1−hΘ(x(t)))
```

- 出力層の誤差

```
δ1<4> = y<i> - a1<4>
出力層の誤差 = トレーニングセットの答え - コスト関数が出した値
```

- 2層目の2個目のノードの誤差

```
δ2<2> = (theta12<3> * δ1<3>) + (theta22<3> * δ2<3>)

2層目の2個目のノードの誤差 =
  (theta[1][1] * 3層目の1個目のノードの誤差) +
  (theta[1][2] * 3層目の2個目のノードの誤差)
```

ここで、バイアスの誤差は計算に入れない点に注意。
バイアスの誤差を計算している資料もあるが、バイアスは基本的に必ず1なので
誤差を計算して修正する必要がないから。

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/qc309rdcEea4MxKdJPaTxA_324034f1a3c3a3be8e7c6cfca90d3445_fixx.png?expiry=1490486400000&hmac=u84nJNffSEiv5yL-GWHo7yS0AYHbdQNhvabMjcRI3Jc)


### バックプロパゲーションの実装 Backpropagation in Practice

#### アンロールパラメータ Unrolling Parameters

アンロール = theta、deltaを1つのvectorにまとめる事。
ニューラルネットワークを計算する関数を実行する際、
ほとんどの場合、thetaやdeltaを1つのベクトルに変換して引数で渡す。


thetaはニューラルネットワークを使う場合、行列ごとに持つ事になる。
また、デルタ(誤差)もネットワーク数に応じて増える。
4つの層があるのであれば、1~3まで。

```
L=4
const Theta1, Theta2, Theta3
const D1, D2, D3
```

入力と中間層が10個のノードを持ち、出力は1個のノードの場合
```
s1=10,s2=10,s3=1
```

octaveでの実装例
```
% Theta1~3 & D1~3 = 10x11 metrics(10層x10層+バイアス)
% theta、deltaを1つのvectorにまとめる
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
% 層ごとの値の取り出し
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

#### Gradient Checking

バックプロパゲーションはデバッグしづらい。
小さなバグがあったとしてもコスト関数は最小を目指し、正常に動いてるように見えてしまう。
しかし、その小さなバグがない実装と比べると性能は劣ってしまう。
グラディエントチェッキング(Gradient Checking)というテクニックを使って
その問題を駆逐する方法を学ぶ。


- グラディエントチェッキングのアイディア

凸型のコスト関数グラフがあったとして、その少しずれた左右をとり、その二つを結んだ線を描く。
その線とthetaの傾きは近似しているはず。

```
theta + ε(エプシロン)
theta - ε(エプシロン)
```

- チェック実行手順

1. Dを計算してベクトル化する
2. Dvecとエプシロンを計算した値を比較して小数点2~3桁で近似である事を確認する
3. 確信を持って実際にトレーニング開始する前にそのチェックをはずす事(遅くなる)

octaveでの実装
```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

#### Random Initialization

トレーニングを開始する際にパラメータthetaをランダムに初期化する。
今まで(ロジスティック回帰など)は初期化に0を使っていたが、
__ニューラルネットワークではうまくいかない。__

うまくいかない理由としては、2層目のノードに値を渡す際、
全て0であれば、全てのノードで同じ値になってしまい、誤差も同じになる。
つまり、学習ができない。

octaveでの実装
```
% ここでのINIT_EPSILONはグラディエントチェッキングのエプシロンと無関係！
% ランダム値は-INIT_EPSILON ~ +INIT_EPSILONの範囲におさまる
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.
Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

#### Putting it Together

全てを踏まえたニューラルネットワークの実装方法

- ニューラルネットワークアーキテクチャの選択
入力層の数はトレーニングデータの特徴数
出力層の数は回帰であれば1個、分類であればその分類数
中間層は基本的に1層、中間層を増やすのであれば、中間層のノード数は同じにする。
中間層の数は精度と時間的なコストを考えて剪定する。

- ニューラルネットワークのトレーニング手順
1. weights(theta)を0付近の小さな値でランダム初期化する。
2. フォワードプロパゲーションの実装
3. コスト関数J(Θ)の実装
4. バックプロパゲーションの実装
5. グラディエントチェッキングの実装、確認、チェックをはずす。
6. 最急降下法や最適化関数などを使用して、thetaを最小にする。


```
for i = 1:m,
  Perform forward propagation and backpropagation using example (x(i),y(i))
  (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

![図](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hGk18LsaEea7TQ6MHcgMPA_8de173808f362583eb39cdd0c89ef43e_Screen-Shot-2016-12-05-at-10.40.35-AM.png?expiry=1490486400000&hmac=seM8U1nt9bTbbtuMmXQ8g53ZHndlwDB-62ag84mpYxc)

### Application of Neural Networks

[自動運転の資料動画](https://www.coursera.org/learn/machine-learning/lecture/zYS8T/autonomous-driving)


### クイズ

1 ベクトル化の実装 Δ(L):=Δ(L)+δ(L+1)∗(a(L))T

2 𝚛𝚎𝚜𝚑𝚊𝚙𝚎(𝚝𝚑𝚎𝚝𝚊𝚅𝚎𝚌(𝟷𝟼:𝟹𝟿),𝟺,𝟼)

3 9.0003

4
- 遅くなるので、本格的なトレーニングの前にGradient Checkingを無効にする
- Gradient Checkingはバックプロパゲーションにバグがないか調べるもの

5
- コスト関数の増加は学習率αが大きすぎる
- コスト関数の減少を確認する

## week6

[lecture-slides10 pdf]()

### 機械学習のデバッグ

線形回帰で家の価格予想で、予測値が大きく外れていた場合のデバッグ方法

- より多くのデータセットで学習する（そこまで良くならない場合があり、かなりの時間を割くべきではない理由を今後のビデオで解説する）
- より少ない特徴で学習し、オーバーフィッティングを避ける
- より多くの特徴で学習し、アンダーフィッティングを避ける
- フィーチャーマッピングでより多くの多項式に変更する
- ラムダの値を調整

機械学習診断で、これらのどれが問題かを診断する。(ml diagnostic)
ただ、その診断方法の理解と実装にも時間はかかるが、適当にデバッグするより最終的には有用な手段である。

### 仮説(目的関数)の評価

仮説のデバッグ

- プロットしてオーバーフィッティングか、アンダーフィッティングか確認
次元が多いと可視化難しい(不可能)
- トレーニングセットをトレーニング用とテスト用に分けて最後にテストする。割合は7:3とか
この場合、トレーニングセットのコストが低くなり、テストデータでは高くなる

誤判別の誤差、ゼロワン誤判別の誤差(misclasification error)
error関数は仮説がミスった時に1を返す関数

```
err(hΘ(x),y)
  = 1 if hΘ(x)≥0.5 and y=0
  = 1 if hΘ(x)<0.5 and y=1
  = 0 otherwise
```

エラー率の定義

```
Test Error=1mtest∑mtesti=1err(hΘ(x(i)test),y(i)test)
```

### 特徴と多項式、正規化パラメータの選び方
(翻訳がかなりズレてる。内容理解出来てるか不明)

下記を選びたい。モデル選択問題とも呼ばれる。

- 特徴
- 多項式
- 正規化パラメータ

オーバーフィッティングは過剰な決定境界が引かれることでトレーニングセットにはフィットするがテストセットではフィットしない。トレーニングセットでコストが下がっても必ずしも良い仮説かはわからない。

トレイン(トレーニング)バリデーションという手法

線形関数を選びたい。二次関数から10次関数(10乗の多項式まで)

```
d = 仮説関数の多項式の次元数
```

この多項式の次元数もthetaとしてパラメータに渡すような方法を用いる。(多項式の次元数の選択)
次元数の選択にはクロスバリデーションセットを用い、得られた次元をもとにトレーニングセットで学習。
学習して得られたパラメータからテストセットでの誤差を算出して比較する。

- トレーニングセット6
- クロスバリデーションセット(cv)2
- テストセット2

1. Optimize the parameters in Θ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with Jtest(Θ(d)), (d = theta from polynomial with lower error);




----------


----------------------------------------
# Tips

## 手法

- 分類 (Classification)

  + SVM (サポートベクトルマシン、線形サポートベクトルマシン)
汎化性能が高く、カーネル関数を選択できるのでさまざまなデータに対応できます。

  + K 近傍法
単純なわりに高い精度を誇ります。

  + ランダムフォレスト
過学習を考慮しなくてよい、並列計算しやすいといった特長があります。

- 回帰 (Regression)

  + 回帰
普通の線形回帰です。

  + ラッソ回帰
少ない変数でモデルを作るが、使わない変数があることを仮定しています。

  + リッジ回帰
多重共線性の影響を受けにくく、ラッソ回帰より変数選択力が弱いという特長があります。

  + SVR
カーネルで非線形性を取りこむことができます。

- クラスタリング

  + K 平均法 (KMeans)
クラスタの数を k 個とあらかじめ指定する代表的なクラスタリング手法です。単純で高速に動作します。

  + 混合ガウス分布 (GMM)
クラスタの所属確率を求めることができます。正規分布を仮定します。

  + 平均変位法 (MeanShift)

カーネル密度推定を用いたロバストでノンパラメトリックな手法です。設定するカーネル幅 (半径 h ) によって自動的にクラスタの数が決まります。入力点群すべてに対して最急降下法の原理を用いて半径 h の円を考え中心点を計算するのでコストが高くなりがちです。
画像のセグメンテーション、エッジ保存の画像の平滑化といった場面にも応用される手法です。
カーネル幅 h を無限大にしたミーンシフトクラスタ解析が k 平均法であると解釈することもできます。

- 次元削減 (Dimensional Reduction)

  + 主成分分析 (PCA)
疎行列も扱え速いという特長があります。正規分布を仮定します。

  + 非負値行列因子分解 (NMF)
非負行列のみ使えますが、より特徴を抽出しやすいこともあります。
その他に線形判別 (LDA) や Deep Learning なども使えます。

## Anaconda in Python

[参考](http://qiita.com/t2y/items/2a3eb58103e85d8064b6)

- 主要ライブラリをオールインワンでインストール
- condaというパッケージ管理システム入り(pipの代わり)
- condaでバンージョン管理も可(pyenvの代わり)
- condaで仮想環境管理も可(virtualenv/venの代わり)


1. install pyenv
2. install python environments

```
pyenv install anaconda3-4.2.0 = python 3.6
pyenv install anaconda3-2.5.0 = python 3.5
pyenv install anaconda2-4.2.0 = python 2.7
```

3. install python environments

pyenv global anaconda2-4.2.0
```
$ pyenv global anaconda3-2.5.0
Python 3.5.1 :: Anaconda 2.5.0 (x86_64)
$ pip list | grep numpy      # condaでパッケージ管理するのでpip入ってはいるけどconda通したほうが良い
numpy (1.11.3)
numpydoc (0.6.0)
$ /anaconda/bin/conda list | grep numpy
numpy                     1.11.3                   py27_0
numpydoc                  0.6.0                    py27_0
```

## OpenCV cv2

- 画像や動画を加工するpythonのライブラリ
- 線画にしたり顔を検出したり？
- 機械学習では、CNN(畳み込みニューラルネットワーク)で、白黒や輪郭情報などの学習に必要な要素にフィルタしてデータを最適化する。

```
# anaconda環境前提
anaconda3(python3.6)が現状対応してない。python~3.5まで。。
$ conda install -c https://conda.anaconda.org/menpo opencv3
$ python
>>> import cv2
>>> cv2.__version__
'3.1.0'
```

## TensorFlow

### 用例

[画像分類](https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet)
[画像分類 inception](https://github.com/tensorflow/models/tree/master/inception/inception)

[画像キャプション](https://github.com/tensorflow/models/tree/master/im2txt)

[RNN](https://github.com/tensorflow/models/tree/master/tutorials/rnn)
[教師なし学習](https://github.com/tensorflow/models/tree/master/tutorials/embedding)

[自然言語CNN 畳み込みニューラルネットワーク](https://github.com/dennybritz/cnn-text-classification-tf)
[自然言語のベクトル化、計算(word2vec)](https://deepage.net/bigdata/machine_learning/2016/09/02/word2vec_power_of_word_vector.html)

[強化学習DQN(keras-rl)](https://github.com/matthiasplappert/keras-rl)
[強化学習DQN(OpenAiGym)](https://github.com/openai/gym)
[強化学習DQN(OpenAiUniverse)](https://github.com/openai/universe)

[追加学習(sequential learning)](http://wired.jp/2017/03/27/deepmind-sequential-memory/)


### install

詳しくは[ここ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#anaconda-installation)
[TF_PYTHON_URL](https://www.tensorflow.org/install/install_mac#TF_PYTHON_URL)

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
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl # githubを参照！！ Mac OS X, CPU only, Python 3.4~

$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0rc1-py3-none-any.whl # tensorflow v1.1.0rc1

$ pip install --ignore-installed --upgrade $TF_BINARY_URL
$ python
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

## sonnet

```
$ brew upgrade bazel
$ bazel version
Build label: 0.4.5-homebrew

$ git clone --recursive https://github.com/deepmind/sonnet
$ cd sonnet/tensorflow
$ ./configure
$ cd ../

$ mkdir /tmp/sonnet
$ bazel build --config=opt :install
$ ./bazel-bin/install /tmp/sonnet

```





----------------------------------------
# TensorFlowで学ぶディープラーニング入門

## TensorFlow入門

### 考え方

- 与えられたデータを元にして道のデータを予測する数式を考える

    y = w0 + w1x + w2x2 + w3x3 + w4x4

- 数式に含まれるパラメータの良し悪しを判断する誤差関数を用意する

    E = 1/2*Σ12,n=1(yn - tn) ** 2

- 誤差関数を最小にするようにパラメータの値を決定する(tensorflowの役目)

### ニューラルネットワーク

複数の仮説関数(ノード)にパラメータを通して得られた結果をパラメータとして、
さらに仮説関数に通していく事で結果をより正しく構成していく。
1層のノード数を増やしたり、層自体を増やしたりする。

[neural-network-zoo](http://postd.cc/neural-network-zoo/)
![neuralnetworks](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)

[図](https://qiita-image-store.s3.amazonaws.com/0/52867/2e9812a1-7ab2-5d49-45fd-5070a4a9724f.jpeg)
[図](https://qiita-image-store.s3.amazonaws.com/0/52867/04b0453e-7486-d225-e492-d86ef2bfe4cf.jpeg)

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

## パーセプトロン

and、or、xorなどを組み合わせたニューラルネットワークの前身
パーセプトロンは活性化関数にステップ関数を使用している。

```
# ステップ関数
a = b + w1x1 + w2x2
# 活性化関数
y = h(a)
```

## ニューラルネットワーク

h(x) = 目的関数 = 活性化関数(activation function)
ステップ関数 = 階段関数 = 閾値を界に出力が切り替わる関数 =

ニューラルネットワークも活性化関数にステップ関数以外の関数を使用している。
下記はシグモイド関数を使用する例

```
# シグモイド関数
h(x) = 1 / 1 + exp(-x)
```

ステップ関数はプロットすると階段のように、0か1の値が返される。
シグモイド関数は0か1ではなく、少数を含むなめらかな勾配の値が返される。

### 目的関数(活性化関数)の種類

#### 中間層の活性化関数

- ステップ関数 = 0 or 1
- シグモイド関数 = なめらかな曲線
- ReLU関数 = 0以下なら0、0以上ならそのまま

#### 出力層の活性化関数

- 恒等関数 = そのままreturn = 回帰で利用
- シグモイド関数 = なめらかな曲線 = 2値分類で利用
- ソフトマックス関数 = 他クラス分類で利用

### ニューラルネットワークの実装

```
# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 恒等関数(回帰における出力層の活性化関数)
def identity_function(x):
    return x

# ソフトマックス関数(他クラス分類における出力層の活性化関数)
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

# ニューラルネットワーク
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    ... const b1,W2,b2,W3,b3
    return network

# フォワードプロパゲーション
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3

    y = identity_function(a3) # 回帰の場合
    # y = softmax(a3) # 他クラス分類の場合

    return y

network = init_network()
x = np.array([1,0, 0.5])
y = forward(network, x)
```

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

----------------------------------------

# 参考

- [neural-network-zoo](http://postd.cc/neural-network-zoo/)
- [word2vec](https://deepage.net/bigdata/machine_learning/2016/09/02/word2vec_power_of_word_vector.html)
- [octave to numpy](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html)
