# coding: utf-8
import sys
import os
from json import dumps
from bottle import Bottle, response

root_app = Bottle()

# loader
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root + "/app")
sys.path.append(root + "/app/lib")

# load apps
from inception import inception_app
root_app.merge(inception_app)

@root_app.route('/')
def index():
    response.content_type = 'application/json'
    return dumps({
        '/': {
            'GET': {
                'params': {},
                'description': 'show api routing'
            }
        },
        '/inception': {
            'GET':  {
                'params': {
                    'url': 'image url (jpg)'
                },
                'description': 'classify image url'
            },
            'POST': {
                'params': {
                    'image': 'image (binary)'
                },
                'description': 'classify image binary'
            }
        }
    })

root_app.run(host='0.0.0.0', port=8080, debug=True)