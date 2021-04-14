# import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
import time
import pandas as pd

# function to extract news for stock ticker
def news_crawler(base_url, tickers):
    news_tables = {}

    for ticker in tqdm(tickers):
        url = base_url + ticker

        req = Request(url=url, headers={'user-agent': 'my-scraper/0.1'})
        response = urlopen(req)
        html = BeautifulSoup(response, 'lxml')

        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    return news_tables


# function to extract share price for a company for the last 7 days
def historical_price(url, tickers):
    # setting up chrome driver
    driver = webdriver.Chrome()

    historical_price_data = []

    # loop through the list of tickers
    for ticker in tqdm(tickers):
        driver.get(url)
        time.sleep(1)

        # Enter name of company in searchbox, and wait for 2 seconds
        driver.find_element_by_xpath("//input[@placeholder = 'Search for news, symbols or companies']").send_keys(
            ticker)
        time.sleep(2)

        # Click on Search icon and wait for 2 seconds
        driver.find_element_by_xpath("//button[@id= 'header-desktop-search-button']").click()
        time.sleep(2)

        # Driver clicks on Historical Data tab and sleeps for 3 seconds
        driver.find_element_by_xpath("//span[text() = 'Historical Data']").click()
        time.sleep(3)

        driver.execute_script("window.scrollBy(0,100)")
        time.sleep(2)

        webpage = driver.page_source

        # start scraping the historical price data
        htmlpage = BeautifulSoup(webpage, 'lxml')
        table = htmlpage.find('table', class_='W(100%) M(0)')
        rows = table.find_all('tr', class_='BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)')

        for i in range(0, len(rows)):
            try:
                row_dict = {}
                values = rows[i].find_all('td')
                if len(values) == 7:
                    row_dict["Ticker"] = ticker
                    row_dict["Date"] = values[0].find('span').text.replace(',', '')
                    row_dict["Open"] = values[1].find('span').text.replace(',', '')
                    row_dict["High"] = values[2].find('span').text.replace(',', '')
                    row_dict["Low"] = values[3].find('span').text.replace(',', '')
                    row_dict["Close"] = values[4].find('span').text.replace(',', '')
                    row_dict["Adj Close"] = values[5].find('span').text.replace(',', '')
                    row_dict["Volumn"] = values[6].find('span').text.replace(',', '')
                historical_price_data.append(row_dict)
            except:
                print("Row number: " + str(i))
            finally:
                i = i + 1

    driver.quit()
    df_hist_price = pd.DataFrame(historical_price_data)

    return df_hist_price
