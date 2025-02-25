import pandas as pd

# load the data
filename='kopken_cleaned.csv'
file_path = f'data/{filename}'
df = pd.read_csv(file_path)

#-----------------------------------------
# sentiment analysis using vaderSentiment
#-----------------------------------------
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

# save the labeled data
filename_saved = 'kopken_labeled.csv'
file_path_saved = f'data/{filename_saved}'
df.to_csv(file_path_saved, index=False)