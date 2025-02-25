import pandas as pd

df = pd.read_csv('../data/output/tokens.csv')

from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Text'].apply(get_sentiment)