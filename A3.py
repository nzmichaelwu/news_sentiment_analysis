# Import libraries
from webcrawler import news_crawler
from webcrawler import historical_price
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# set options
pd.set_option('mode.chained_assignment', None)
pd.options.display.max_colwidth = 200

# initialise variables
base_url = 'https://finviz.com/quote.ashx?t='
yahoo_finance = 'https://finance.yahoo.com/'

# top 10 tech stocks in US, most of them would have been impacted during Covid one way or another
tickers = ['AMZN', 'AAPL', 'MSFT', 'INTC', 'AMD', 'DIS', 'NFLX', 'FB', 'TWTR', 'GOOG']

# need to split the ticker list into 2 because yahoo finance has a limit on scraping in one session
tickers_list_1 = ['AMZN', 'AAPL', 'MSFT', 'INTC', 'AMD']
tickers_list_2 = ['DIS', 'NFLX', 'FB', 'TWTR', 'GOOG']

stop_words = set(stopwords.words('english'))

news_tables = {}

# get news headlines
news_tables = news_crawler(base_url, tickers)

# get historical price
hist_price_1 = historical_price(yahoo_finance, tickers_list_1)
hist_price_2 = historical_price(yahoo_finance, tickers_list_2)

df_hist_price = pd.concat([hist_price_1, hist_price_2]) \
    .reset_index(drop=True) \
    .dropna()

cols = df_hist_price.columns.drop(['Ticker', 'Date'])
df_hist_price[cols] = df_hist_price[cols].apply(pd.to_numeric, errors='coerce')


# extract news
def extract_news(news_tables):
    extracted_news = []

    for file_name, news_table in news_tables.items():
        # iterate through all <tr> tag in news_table
        for news in news_table.findAll('tr'):
            text = news.a.get_text()  # get text from tag <a>
            date_scrape = news.td.text.split()  # get date from tag <td>

            # if length of 'date_scrape' is 1, load 'time as the only element
            if len(date_scrape) == 1:
                time = date_scrape[0]
            # else load date as the first element and time as the second element
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]

            extracted_news.append([ticker, date, time, text])

    return extracted_news


extracted_news = extract_news(news_tables)

df_news = pd.DataFrame(extracted_news, columns=['ticker', 'date', 'time', 'text'])

# quick insights on news headlines gathered
num_news_gathered = len(df_news)

df_news['word_length'] = [len(word) for word in df_news['text']]
x = df_news['word_length']
plt.hist(x)


# function to pre-process news
def preprocess_news_text(news):
    # initialise lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    news = news.lower()

    # remove punctuations
    news = re.sub('[^a-zA-Z]', ' ', news)

    # remove stopwords and lemmatization
    news_words = word_tokenize(news)
    news_words = [wordnet_lemmatizer.lemmatize(word) for word in news_words if word not in stop_words]

    news_lemmed = ' '.join(news_words)
    return news_lemmed


df_news['news_cleaned'] = df_news['text'].apply(preprocess_news_text)


# function to convert string to datetime
def to_date(dates, date_format):
    date_list = dates.tolist()
    date_list_converted = []
    for date in date_list:
        date_con = datetime.datetime.strptime(date, date_format)
        date_list_converted.append(date_con)

    converted_dates = pd.Series(date_list_converted)
    return converted_dates


df_news['date'] = to_date(df_news['date'], '%b-%d-%y')
list_of_dates = df_news['date'].unique()
df_hist_price['Date'] = to_date(df_hist_price['Date'], '%b %d %Y')
df_hist_price_mod = df_hist_price[df_hist_price['Date'].isin(list_of_dates)]

# NLP task 1 - sentiment analysis
sid = SentimentIntensityAnalyzer()

sentiment = df_news.apply(lambda r: sid.polarity_scores(r['news_cleaned']), axis=1)
df_sent = pd.DataFrame(list(sentiment))
df_news_sentiment = df_news[['ticker', 'date', 'time', 'text', 'news_cleaned']].join(df_sent)


def get_sentiment(compound):
    if compound < 0:
        return "Negative"
    elif compound == 0:
        return "Neutral"
    else:
        return "Positive"


df_news_sentiment['sentiment'] = df_news_sentiment['compound'].apply(get_sentiment)

# NLP task 2 - information extraction (keyword extraction)
news_corpus = df_news_sentiment['news_cleaned']


# function using TfidfVectorizer to find keywords based on TFIDF score for each news headline in the news_corpus
def extract_keywords(news_corpus):
    vectorizer = TfidfVectorizer(use_idf=True)
    news_vectors = vectorizer.fit_transform(news_corpus)

    keyword_dict = {}
    keywords_list = []
    keyword = []

    # loop through each news headline in the corpus
    for index in tqdm(range(len(news_corpus))):
        vector = news_vectors[index]

        # for all words in the corpus store the score for each word in a dictionary
        for i in range(len(vectorizer.get_feature_names())):
            keyword_dict[vectorizer.get_feature_names()[i]] = round(vector.T.todense()[i].tolist()[0][0], 4)

        # remove words that have 0 score (i.e. not important), sort the dictionary, and add each dictionary of
        # keywords into the list
        keyword_dict = {k: v for k, v in keyword_dict.items() if v != 0.0}
        keyword_dict_sorted = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
        keywords_list.append(keyword_dict_sorted)

    # loop through the keywords_list and convert key and value of each dictionary in the list into a massive string,
    # then append it to a list called keyword.
    for i in range(len(keywords_list)):
        k = ', '.join([item[0] + ": " + str(item[1]) for item in keywords_list[i]])
        keyword.append(k)

    # convert the keyword list into a Pandas dataframe, so that the df will only have one column called keyword,
    # and contains the keyword and the corresponding score for each index in the news_corpus.
    df_keyword = pd.DataFrame(keyword, columns=['keyword'])

    return df_keyword


df_keyword = extract_keywords(news_corpus)

df_sentiment_keyword = pd.concat([df_news_sentiment, df_keyword], axis=1)


# last part - plot sentiment score vs stock price
def fig_plot(df_plot, ticker):
    fig = make_subplots(specs=[[{"secondary_y": True}]]) \
        .add_trace(
        go.Scatter(x=df_plot['date'], y=df_plot['compound'], name="Average Sentiment"),
        secondary_y=False,
    ) \
        .add_trace(
        go.Scatter(x=df_plot['date'], y=df_plot['Adj Close'], name="Adjusted Close"),
        secondary_y=True,
    ) \
        .update_layout(
        title_text="Average Sentiment vs Adjusted Close - " + ticker,
        plot_bgcolor='rgba(0,0,0,0)'
    ) \
        .update_xaxes(title_text="Date") \
        .update_yaxes(title_text="Avg. Sentiment", secondary_y=False) \
        .update_yaxes(title_text="Adj. Close", secondary_y=True)

    fig.show()


for ticker in tickers:
    df_sent = df_news_sentiment[df_news_sentiment['ticker'] == ticker][['ticker', 'date', 'compound']]
    df_sent_avg = df_sent.groupby('date').agg({"compound": "mean"}).reset_index()
    df_hist_price = df_hist_price_mod[(df_hist_price_mod['Ticker'] == ticker) &
                                      (df_hist_price_mod['Date'].isin(df_sent_avg['date']))][
        ['Ticker', 'Date', 'Adj Close']]

    df_plot = pd.merge(df_sent_avg, df_hist_price, left_on='date', right_on='Date', how="left") \
        .drop(columns=['Date']) \
        .fillna({'Ticker': ticker}) \
        .set_index('date') \
        .interpolate(method="time") \
        .reset_index()

    fig_plot(df_plot, ticker)
