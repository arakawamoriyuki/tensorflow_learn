from bottle import route, run, template

@route('/')
def index():
    return "Hello World"

@route('/hello/<name>')
def hello(name):
    return template('<b>Hello {{name}}</b>!', name=name)

run(host='0.0.0.0', port=8080)