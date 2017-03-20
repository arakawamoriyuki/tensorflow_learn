
# j = コスト関数、θ1を求める為のアルゴリズム

# 1/2m( (h(xi) - yi)**2 )
# m = 3, 1/6( (x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2 )

# θ0 = 開始地点
# θ1 = 傾き
# m = データ数
# y = 実データの値(eg. 家の値段)
# x = 入力(eg. 家の広さ)

# θ1を調査する一次関数、実データ
@line = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
# @line = [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
# @line = [[1.0, 0.2], [2.0, 0.4], [3.0, 0.6]]
# @line = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] # @range = 30
# @line = [[1.0, 1.0], [2.0, 0.5], [3.0, 0.0]] # 傾きの調整が必要
# @line = [[2.9, 2.3], [1.0, 0.3], [1.2, 2.3], [2.5, 0.9], [0.4, 2.0], [4.3, 2.2]]

# 実データ個数
@m = @line.size

# コスト関数を使って調査する範囲、傾きに応じて変更する必要あり。大きいほど正確
@range = 20

def j theta_1

  line = @line.dup
  line.unshift(result = 0.0)

  # right = (h(xi) - yi)**2
  right = line.reduce do |result, point|
    x = point[0]
    y = point[1]

    # (h(xi) - yi)**2
    result += ((theta_1 * x) - y) ** 2.0
  end

  # 1/2m
  left = 1.0 / (2.0 * @m)

  return left * right
end

# θ1を仮説で値を入れ、最小の値が正解
min_θ1 = nil
min_j = nil
@range.times do |index|
  θ1 = (index * 0.1).round(2)
  puts "θ1 = #{θ1}, j(θ1) = #{j(θ1)}"

  # 最小値を保存
  min_j = j(θ1) if min_j.nil?
  min_θ1 = θ1 if min_θ1.nil?

  min_θ1, min_j = θ1, j(θ1) if min_j > j(θ1)
end

# 最小である1.0がもっともらしい。θ1 = 1.0
puts "minimize θ1 = #{min_θ1}"

