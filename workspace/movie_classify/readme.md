# youtube-8m-classify

|file path|description|
|:-:|:-:|
|/input/label_names.csv|タグIDとタグ名|
|/input/sample_submission.csv|提出サンプル|
|/input/train_labels.csv|トレーニングデータ|
|/input/validate_labels.csv|テストデータ?|


[kaggle](https://www.kaggle.com/c/youtube8m)
[youtube8m](https://research.google.com/youtube8m/)
[youtube8m github](https://github.com/google/youtube-8m)


```
$ pyenv global anaconda2-4.2.0
$ TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL

$ git clone https://github.com/google/youtube-8m

$ mkdir -p features
$ cd features

# datasets {molecule}/{denominator} download
# {index} = number
# {format} = video or frame
# {dataset_type} = train or test or validate
$ curl data.yt8m.org/download.py | shard={molecule},{denominator} partition={index}/{format}_level/{} mirror=asia python
# eg. 1/1000 index1 video datasets download
$ curl data.yt8m.org/download.py | shard=1,1000 partition=1/video_level/train mirror=asia python

$ MODEL_DIR=/tmp/yt8m
$ python train.py --train_data_pattern='../features/train*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model

$ sudo mkdir $MODEL_DIR/video_level_logistic_model
$ python eval.py --eval_data_pattern='../features/validate*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model --run_once=True
$ tensorboard --logdir=$MODEL_DIR
```

