
- chromedriver download and append path
- gem install selenium-webdriver

```
$ gem install selenium-webdriver
```

```crawler.rb
require './translator'

translate do |translator|
  puts translator.call('translator')
end
```

```
$ ruby crawler.rb
翻訳者
```

<!--
[headless X11 installer](https://support.apple.com/ja-jp/HT201341)

```
$ /opt/X11/bin/Xvfb
```
-->