# Pydata okinawa 2017-02-25

STARTUP CAFE KOZA

## PyData.Okinawaについて

話し手 = 大塚まこと

ゲスト = 吉田尚人
image-to-image 線画から
dlss lss 2017 school
chainer強化学習ライブラリ chainerRL 2/16

## 自己紹介



## 深層強化学習入門

[スライド質問](http://goo.gl/slides/gw3nyk)

話し手 = 吉田尚人 = Groove Xで家庭用ロボット開発してる人


### 強化学習

agent自立的に行動を決定、score(報酬)を最大化する
RL(強化学習) = AI
鳩の例
制御と強化学習の違い 環境がblackboxかどうか

強化学習の使い所 = 入力と出力があって学習データがない

歩き方がわからないロボットに適当に動かさせて学習させる(報酬は動いた距離)

強化学習の種類

- 熟考型 モデルベース RNN シミュレーション 学習は早い アームを動かすなど小規模な環境
- 直感型 モデルフリー DNN すぐアクションを取れる 学習に時間がかかる

強化学習を本当に利用すべきか考える。
株は時系列学習、教師あり学習の方がいい。

シミュレータはほぼ必須。ロボットは壊れる。

マルコフ決定過程
行動と前回の状態の利用。RNNみたいなもん?

自動運転などは人間のデータで学習させた後に強化学習をする。最初から強化学習は難しい。

強化学習パッケージ、最近出てきた。

- chainerRL
- keras-rl
- openai gym
- openai universe

### 深層強化学習

誤差関数で収束する訳ではない？まだまだ実験的。

深層強化学習 = 深層学習 + 強化学習 + a

dqn 直感型

#### 書籍紹介

- ? 4万
- ?
- これからの強化学習
- 強くなるロボティックゲームプレイヤーの作り方

#### まとめ

- 強化学習の使い所は考える
- シミュレータのあるなしで難易度がだいぶ変わる
- モデルフリー、モデルベース

## OpenAI Universeで深層強化学習


```
$ git clone https://github.com/ugo-nama-kun/pydata_okinawa2017
$ cd pydata_okinawa2017
$ pip install -r requirements.txt
$ jupyter notebook
```

http://localhost:8888/notebooks/introduction_deep_rl.ipynb


パッケージ
アルゴリズム
環境







#
