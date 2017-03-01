
- chromedriver download and append path
- gem install selenium-webdriver

```
$ gem install selenium-webdriver
```

```crawler.rb
require './image-collector'

SEARCH = ARGV[0] || '犬'

image_collector SEARCH do |save, index|
  puts "#{dest}/#{index-1}.jpg"
  save.call("#{dest}/#{index-1}.jpg")
end
```

```
$ ruby crawler.rb トラック
```

<!--
[headless X11 installer](https://support.apple.com/ja-jp/HT201341)

```
$ /opt/X11/bin/Xvfb
```
-->