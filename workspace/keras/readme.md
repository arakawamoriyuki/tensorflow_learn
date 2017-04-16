
# keras in TensorFlow 1.1.0-rc1(Pre-release)

```
$ python --version
Python 3.5.3 :: Anaconda 2.5.0 (x86_64)

<!-- # 本体
$ sudo pip install keras
$ conda install h5py
# 可視化
$ brew install graphviz
$ pip install pydot -->

# tensorflow v1.1 install
$ pip uninstall tensorflow
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0rc1-py3-none-any.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL

$ python
>>> import tensorflow as tf
>>> tf.__version__
'1.1.0-rc1'
```

[keras documents](https://keras.io/ja)

[tensorflow playground](http://playground.tensorflow.org)

![活性化関数](https://cdn-ak.f.st-hatena.com/images/fotolife/i/imslotter/20170107/20170107153956.png)

[参考](http://www.procrasist.com/entry/2017/01/07/154441)