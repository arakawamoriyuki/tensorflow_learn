require './image-collector'

SEARCH = ARGV[0] || '犬'

dest = 'images'

image_collector SEARCH do |save, index|
  puts "#{dest}/#{index-1}.jpg"
  save.call("#{dest}/#{index-1}.jpg")
end