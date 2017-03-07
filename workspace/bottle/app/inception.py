# coding: utf-8
from bottle import Bottle, request, response
from json import dumps

import requests

from inception_v3 import run_inference_on_image

TMP_IMAGE = '/tmp/inception_tmp.jpg'

inception_app = Bottle()


# GETで画像urlを渡して分類を判定する。ContentType:multipart/form-data
@inception_app.get('/inception')
def get_inception():
    # request image url
    response = requests.get(request.params.image, stream=True)

    # save image
    with open(TMP_IMAGE, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # classify image
    result = run_inference_on_image(TMP_IMAGE)

    response.content_type = 'application/json'
    return dumps(result)


# POSTで画像binaryを渡して分類を判定する。ContentType:multipart/form-data
@inception_app.post('/inception')
def post_inception():
    # request image binary
    image = request.files.get('image')

    # save image
    image.save(TMP_IMAGE, overwrite=True, chunk_size=1024)

    # classify image
    result = run_inference_on_image(TMP_IMAGE)

    response.content_type = 'application/json'
    return dumps(result)


# apiテスト用View
@inception_app.get('/inception/test')
def get_inception_test():
    return '''
<form action="/inception" method="post" enctype="multipart/form-data">
    <input type="submit">
    <input type="file" name="image">
</form>
'''
