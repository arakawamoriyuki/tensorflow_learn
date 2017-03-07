# ml_bottle

- 超軽量フレームワーク[bottle](http://bottlepy.org/docs/dev/index.html)に乗せた機械学習環境とAPI群

## hello world

```server.py
from bottle import route, run, template

@route('/hello/<name>')
def index(name):
    return template('<b>Hello {{name}}</b>!', name=name)

run(host='0.0.0.0', port=8080)
```

```
$ python server.py
http://localhost/hello/your_name
```

## 動作環境

```
$ python --version
Python 2.7.13 :: Anaconda 4.3.0 (x86_64)
$ python
>>> import tensorflow as tf
>>> tf.__version__
'1.0.0'
>>> import bottle
>>> bottle.__version__
'0.12.13'
```

## endpoints

|endpoint|method|description|
|:-:|:-:|:-:|
|/|GET|ルーティング一覧|
|/inception|GET|パラメータimageに画像urlを乗せて分類|
|/inception|POST|パラメータimageに画像binaryを乗せて分類|
|/inception/test|GET|inception apiを試すview|

## local run

localで起動する場合、下記環境が必要になります。(dockerで良ければlocal環境は必要ありません)

- bottle
- requests
- numpy
- six
- tensorflow

bottleとrequestsはpipでinstallできます。
numpyやsixなどは機械学習環境をまとめた[anaconda](https://www.continuum.io/downloads)で環境を作ります。
tensorflowは[github](https://github.com/tensorflow/tensorflow)に従って環境を作ります。

```
$ python server.py
```

## background local run

### run
```
$ python server.py > /dev/null 2>&1 &
```

### kill
```
$ ps | grep python server.py
$ kill -9 {pid}
```

## docker run

- linux (Linux cd4ac837b33e 4.4.14-moby #1 SMP Wed Jun 29 10:00:58 UTC 2016 x86_64 GNU/Linux)
- anaconda (Python 3.6.0 :: Anaconda 4.3.0 (64-bit))
- tensorflow (1.0.0)
- bottle (0.12.13)

### build and run

```
$ docker build --rm -t anaconda-bottle .
$ docker run -d -p 8080:8080 --name anaconda-bottle anaconda-bottle
```

### login

```
$ docker exec -it anaconda-bottle bash
```







#