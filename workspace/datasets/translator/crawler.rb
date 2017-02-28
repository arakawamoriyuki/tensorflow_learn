require 'selenium-webdriver'

src_file = 'imagenet_synset_to_human_label_map.txt'
dest_file = 'dest.txt'
separator = '\t'
SLEEP = 3
SLEEP_FINDRESULT = 3
BATCH_SIZE = 10

def translate_setup
  driver = Selenium::WebDriver.for :chrome
  wait = Selenium::WebDriver::Wait.new(:timeout => 10)
  driver.navigate.to 'https://translate.google.co.jp/#en/ja'
  yield lambda{|english|
    wait.until { driver.find_element(css: "#source") }.send_keys(english)
    wait.until { driver.find_element(css: "#gt-submit") }.click
    translated = wait.until {
      elements = nil
      while true
        sleep SLEEP_FINDRESULT
        elements = driver.find_elements(css: "#result_box > span")
        break if BATCH_SIZE <= elements.count
      end
      driver.find_elements(css: "#result_box > span").map { |element| element.text }.join("\n")
    }
    wait.until { driver.find_element(css: "#source") }.clear
    sleep SLEEP
    return translated
  }
  driver.quit
end

File.open(dest_file) do |dest|
  saved_count = dest.count
  File.open(dest_file, 'a') do |dest|
    File.open(src_file) do |src|
      translate_setup do |translator|
        src.each_slice(BATCH_SIZE).to_a.each_with_index do |batch_lines, batch_index|
          if saved_count <= (batch_index * BATCH_SIZE)
            labels = batch_lines.map {|line|
              label = line.split(/#{separator}/)[1].strip
            }
            translateds = translator.call(labels.join("\n")).split("\n")
            translateds.each_with_index do |translated, index|
              label = labels[index]
              index += (batch_index * BATCH_SIZE) + 1
              puts "#{index}: #{label} -> #{translated}"
              dest.puts([label, translated].join("\t"))
            end
          end
        end
      end
    end
  end
end