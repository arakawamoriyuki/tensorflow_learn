require './image-collector'

SEARCH = ARGV[0] || '犬'

dest = 'images'

image_collector SEARCH do |blob, index|
  puts "#{dest}/#{index}.jpg"
  File.open("#{dest}/#{index}.jpg", 'wb') do |file|
    file.write blob
  end
end