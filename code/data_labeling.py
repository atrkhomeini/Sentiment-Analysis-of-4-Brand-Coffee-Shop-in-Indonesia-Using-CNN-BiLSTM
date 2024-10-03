import pandas as pd

# load the data
df = pd.read_csv('data/fore_cleaned.csv')

# sentiment analysis using vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# applying sentiment analysis
df['sentiment'] = df['full_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

#label sentiment
## compound score ranges from -1 to 1
def categorize_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)

# sentiment analysis using Transformers
from transformers import pipeline
# Load pre-trained model for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# analyze sentiment
df['sentiment_transformers'] = df['full_text'].apply(lambda x: sentiment_analyzer(x))
