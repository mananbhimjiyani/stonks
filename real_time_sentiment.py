from gnews import GNews
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata


def get_current_sentiment(stock_symbol):
    google_news = GNews(language='en', country='IN', period='24h', start_date=None, end_date=None, max_results=1)
    data = google_news.get_news('stock_symbol')
    df = pd.DataFrame(data)
    df.drop(['publisher','url','title','published date'],axis=1,inplace = True)


    # stock_symbol = "TCS"
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1d", interval="1m")
    latest_closing_price = data['Close'].iloc[-1]
    df['Prices'] = latest_closing_price


    df['Comp'] = 0.0
    df['negative'] = 0.0
    df['Neutral'] = 0.0
    df['Positive'] = 0.0

    ccdata = df.copy()



    sentiment_i_a = SentimentIntensityAnalyzer()

    tempcomp = 0
    sentiment_i_a = SentimentIntensityAnalyzer()
    for index, row in ccdata.iterrows():
        try:
            sentence_i = unicodedata.normalize('NFKD', row['description'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            ccdata.at[index, 'Comp'] = sentence_sentiment['compound']
            ccdata.at[index, 'Negative'] = sentence_sentiment['neg']
            ccdata.at[index, 'Neutral'] = sentence_sentiment['neu']
            ccdata.at[index, 'Positive'] = sentence_sentiment['pos']
            tempcomp = ccdata.at[index,'Comp']
        except TypeError:
            print(ccdata.at[index, 'Tdescription'])
            print(index)
    
    return tempcomp


def get_current_closing(stock_symbol):
    #stock_symbol = "UPL.NS"
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1d", interval="1m")
    # print(data)
    # if(data == NULL):
    #     try:

    latest_closing_price = data['Close'].iloc[-1]
    return latest_closing_price