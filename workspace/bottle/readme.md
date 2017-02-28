# bottle

- 超軽量フレームワーク[bottle](http://bottlepy.org/docs/dev/index.html)

## localで起動

```
$ pip install bottle
$ python server.py
```

## localでログを捨てつつバックグラウンド実行

- run
```
$ python server.py > /dev/null 2>&1 &
```

- kill
```
$ ps | grep python server.py
$ kill -9 {pid}
```

## dockerで起動

- build and run
```
$ docker build --rm -t anaconda-bottle .
$ docker run -d -p 8080:8080 --name anaconda-bottle anaconda-bottle
```

- login
```
$ docker exec -it anaconda-bottle bash
```







#