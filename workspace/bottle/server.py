# coding: utf-8
import sys
import os
from json import dumps

from bottle import Bottle, response, route

# loader
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root + "/app")
sys.path.append(root + "/lib")

# load apps
from inception import inception_app

root_app = Bottle()
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
        '/inception/test': {
            'GET':  {
                'params': {},
                'description': 'classify image test view'
            }
        },
        '/inception': {
            'GET':  {
                'params': {
                    'image': 'image url (jpg)'
                },
                'description': 'image url classify api'
            },
            'POST': {
                'params': {
                    'image': 'image binary (jpg)'
                },
                'description': 'image binary classify api'
            }
        }
    })

root_app.run(host='0.0.0.0', port=8080, debug=True)