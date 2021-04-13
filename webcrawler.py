# import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from tqdm import tqdm

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
