import pandas as pd
from transformers import pipeline
# Correct model identifier
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

# Initialize the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model=model_name)

#-----------------------------------------------------------------------------
#load data tokens
#-----------------------------------------------------------------------------
df = pd.read_csv('../research/tokens_normalized.csv')

df['Sentiment'] = df['Normalized_Text_Slang'].apply(lambda x: classifier(x)[0]['label'])

print("Sentiment analysis completed successfully!")

df.to_csv('../research/df_sentiment.csv', index=False)