require 'selenium-webdriver'
require 'base64'

SLEEP = 3

def image_collector search
  driver = Selenium::WebDriver.for :chrome
  wait = Selenium::WebDriver::Wait.new(:timeout => 10)
  driver.navigate.to "https://www.google.co.jp/search?q=#{search}&tbm=isch"
  (1..Float::INFINITY).each do |nth_index|
    sleep SLEEP
    image_element = wait.until {
      driver.find_element(css: "#rg_s > .rg_di.rg_bx.rg_el.ivg-i:nth-child(#{nth_index}) img")
    }
    src = image_element.attribute('src')
    blob = Base64.decode64(src.gsub(/^.*,/, ''))
    file_type = src.match(/^.*:(.*);.*$/)[1]
    yield blob, nth_index - 1, file_type
  end
  driver.quit
end