require './translator'

src_file = 'imagenet_synset_to_human_label_map.txt'
dest_file = 'dest.txt'
separator = '\t'
split_index = 1
batch_size = 10

File.open(dest_file, 'a') do |dest|
  saved_count = File.read(dest_file).count("\n")
  File.open(src_file) do |src|
    translate do |translator|
      src.each_slice(batch_size).to_a.each_with_index do |batch_lines, batch_index|
        if saved_count <= (batch_index * batch_size)
          labels = batch_lines.map {|line|
            label = line.split(/#{separator}/)[split_index].strip
          }
          translateds = translator.call(labels.join("\n")).split("\n")
          translateds.each_with_index do |translated, index|
            label = labels[index]
            index += (batch_index * batch_size) + 1
            puts "#{index}: #{label} -> #{translated}"
            dest.puts([label, translated].join("\t"))
          end
        end
      end
    end
  end
end
