import pandas as pd
from transformers import pipeline
# Correct model identifier
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

# Initialize the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model=model_name)

#-----------------------------------------------------------------------------
#load data tokens
#-----------------------------------------------------------------------------
df = pd.read_csv('../../data/output/tokens.csv')

df['Sentiment'] = df['Text'].apply(lambda x: classifier(x)[0]['label'])

print("Sentiment analysis completed successfully!")

#------------------------------------------------------------
# Load data tokens and stopwords removed
#------------------------------------------------------------
df_stopwords = pd.read_csv('../../data/output/tokens_no_stopword.csv')
df_stopwords['Sentiment'] = df_stopwords['Text'].apply(lambda x: classifier(x)[0]['label'])
print("Sentiment analysis completed successfully!")

#------------------------------------------------------------
# visualize the data
#------------------------------------------------------------

print(df['Sentiment'].value_counts())
print(df_stopwords['Sentiment'].value_counts())