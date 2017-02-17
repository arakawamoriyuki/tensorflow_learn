# 標準偏差の計算

require 'complex'
avg = 60
player_length = 10
a_points = [0,5,10,70,80,80,82,85,93,95]
b_points = [50,52,54,60,60,60,61,61,70,72]
tests = {a: a_points, b: b_points}
tests.each do |type, points|
  points.unshift(result = 0)
  # 分散値の計算
  dispersion = points.reduce do |result, point|
    result += ((point - avg) ** 2) * 1.0 # *1.0 = cast float
  end
  dispersion /= player_length

  puts "#{type} #{dispersion}"

  # 標準偏差の計算
  deviation = Math.sqrt(dispersion)
  puts "#{type} #{deviation}"
end