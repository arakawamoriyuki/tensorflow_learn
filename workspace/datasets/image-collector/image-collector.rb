require 'selenium-webdriver'
require 'base64'
require 'net/http'
require 'open-uri'

SLEEP = 1
RETRY_SLEEP = 0.1

def image_collector search
  driver = Selenium::WebDriver.for :chrome
  wait = Selenium::WebDriver::Wait.new(:timeout => 10)
  driver.navigate.to "https://www.google.co.jp/search?q=#{search}&tbm=isch"
  nth_index = 20
  (1..Float::INFINITY).each do |nth_index|
    sleep SLEEP
    image_element = wait.until {
      while nth_index > driver.find_elements(css: "#rg_s > .rg_di.rg_bx.rg_el.ivg-i").count
        sleep RETRY_SLEEP
      end
      driver.find_element(css: "#rg_s > .rg_di.rg_bx.rg_el.ivg-i:nth-child(#{nth_index}) img")
    }
    driver.execute_script('arguments[0].scrollIntoView(true);', image_element)
    src = image_element.attribute('src')
    if (src =~ /^(http[s]?:\/\/.+).*$/) != nil
      open(src) do |string_io|
        blob = string_io.read
        yield lambda { |path|
          File.open(path, 'wb') do |file|
            file.write blob
          end
        }, nth_index
      end
    elsif (src =~ /^.*;base64,.*$/) != nil
      blob = Base64.decode64(src.gsub(/^.*,/, ''))
      yield lambda { |path|
        File.open(path, 'wb') do |file|
          file.write blob
        end
      }, nth_index
    end
    nth_index += 1
  end
  driver.quit
end