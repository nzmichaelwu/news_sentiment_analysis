# Financial News Headline Sentiment Analysis

_Author: Michael Wu, April 2021_

##**Task 1 - Overview**

####_An overview of the issue to be investigated_

Many economists have argued that stock market is random because it is governed by random events based on the efficient market hypothesis. However, researchers have found that it is possible to have a sense of where the stock market will go based on news sentiment around the world. In an increasingly connected global market, news sentiment towards one company may not only indicate its own market performance, but can also be associated with a broader movement on the sentiment and performance of other companies from the same or even different sectors ([Wan, X., Yang, J., Marinov, S. et al., 2021](https://www.nature.com/articles/s41598-021-82338-6)). In short, researchers have found that there is a relationship between news sentiment and stock performance of a company, and that news sentiment of a company could be an indicator of its stock performance in the short term.

The objective of this investigation is to perform sentiment analysis of new headlines for top listed technology companies in US, and extract keywords that contribute to the sentiment for a given news headline. By understanding what keywords in the news headline contribute to the sentiment direction - positive verse negative - of a company, investors and researchers can then use this information to predict the stock performance.

####_How the WebCrawler align to this issue_

In order to achieve the objective of this investigation, we need to obtain news headlines for top listed technology companies in US. To obtain the news headlines, we used a website called FINVIZ, which is a browser-based stock market research platform with market information such as daily news headlines from different news agencies on a company. Given that the key objective of this investigation is to assess the sentiment on news headlines for top listed technology companies in US, a webcrawler on the FINVIZ website to obtain news headlines is directly related to the issue we are investigating.  

In addition, in order to determine whether there is a relationship between share price and news sentiment of a company, we need to obtain historical price of a company. To do so, a webcrawler on the Yahoo Finance website is required to obtain historical price for those top listed technology companies in US.

####_NLP tasks align to the issue_

As mentioned in the objective, researchers have found that there is a relationship between news sentiment and stock performance. And thus, the first NLP task in this investigation is to perform sentiment analysis on company news headlines, as we hypothesise that by understanding the sentiment direction, it can help predict the direction of the stock performance. The second NLP task is an extension to the first, in which we perform semantic analysis and determine keywords that contribute to the sentiment for each news headline. This can help investors identify keywords from news headlines and use that insight to make informed decision on the stock performance in the short term.