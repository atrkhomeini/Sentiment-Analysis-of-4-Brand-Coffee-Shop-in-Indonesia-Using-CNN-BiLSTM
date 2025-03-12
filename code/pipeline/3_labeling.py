import pandas as pd
from transformers import pipeline

#load pretrained model
classifier = pipeline('sentiment-analysis', model="W11wo/indonesian-roberta-large-sentiment-classifier")

#load data
df = pd.read_csv('../../data/output/tokens.csv')

df['Sentiment'] = df['Text'].apply(lambda x: classifier(x)[0]['label'])

print("Sentiment analysis completed successfully!")