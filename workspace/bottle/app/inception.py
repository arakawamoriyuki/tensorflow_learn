# coding: utf-8
from bottle import Bottle, request, response
from json import dumps

import requests

from inception_v3 import run_inference_on_image

inception_app = Bottle()

# @param
@inception_app.route('/inception')
def inception():

    default_image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAoZbaD9CYHo5i4Ez5QGf4uHj5pG_Klwu2oEGTipKg8TRr-6pV'
    source = request.params.url or default_image_url
    output = '/tmp/inception_tmp.jpg'

    response = requests.get(source, stream=True)
    if response.status_code == 200:
        with open(output, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        result = run_inference_on_image(output)
        response.content_type = 'application/json'
        return dumps(result)
    else:
        return dumps({'message':'image url could not be loaded'})


