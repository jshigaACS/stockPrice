from selenium import webdriver
import pandas as pd
from matplotlib import pyplot as plt
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
browser = webdriver.Chrome(
    executable_path='/mnt/c/Users/acs30/Downloads/chromedriver_win32/chromedriver.exe',
    options=options
)

browser.get("https://kabuoji3.com/stock/")
stockSearch=browser.find_element_by_class_name("form_inputs")
stockSearchForm=stockSearch.find_element_by_class_name("form_txt")
stockSearchForm.send_keys("ETF")
btnClick=browser.find_element_by_class_name("btn_submit")
btnClick.click()

stockClick=browser.find_elements_by_class_name("clickable")
print(len(stockClick))