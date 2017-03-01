require 'selenium-webdriver'

TRANSLATE_FROM = 'en'
TRANSLATE_TO = 'ja'
TIMEOUT = 10
SLEEP = 1
RETRY_SLEEP = 0.1

def translate
  driver = Selenium::WebDriver.for :chrome
  wait = Selenium::WebDriver::Wait.new(:timeout => TIMEOUT)
  driver.navigate.to "https://translate.google.co.jp/##{TRANSLATE_FROM}/#{TRANSLATE_TO}"
  yield lambda{|english|
    wait.until { driver.find_element(css: "#source") }.send_keys(english)
    wait.until { driver.find_element(css: "#gt-submit") }.click
    translated = wait.until {
      elements = nil
      while 0 == driver.find_elements(css: "#result_box > span").count
        sleep RETRY_SLEEP
      end
      driver.find_elements(css: "#result_box > span").map { |element| element.text }.join("\n")
    }
    wait.until { driver.find_element(css: "#source") }.clear
    while 0 < driver.find_elements(css: "#result_box > span").count
      sleep RETRY_SLEEP
    end
    sleep SLEEP
    return translated
  }
  driver.quit
end