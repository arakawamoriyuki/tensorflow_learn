
=begin
# 中期試験点数の二乗をフューチャースケーリングと平均正則化を適用する

|中期試験点数(x)|中期試験点数の二乗(x^2)|最終試験点数(y)|
|:-:|:-:|:-:|
|89|7921|96|
|72|5184|74|
|94|8836|87|
|69|4761|78|
=end


x_vector = [7921, 5184, 8836, 4761]

max = 8836.0
min = 4761.0
avg = x_vector.reduce(:+) / x_vector.count # = 6675.0
normalize = max - min # = 4075.0

x_vector.each do |x|
  puts "x(#{x}) => #{(x - avg) / normalize}"
  x
end
