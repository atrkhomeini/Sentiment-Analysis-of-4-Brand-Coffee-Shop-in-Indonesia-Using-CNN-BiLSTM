import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../data/output/elimination.csv')

# -----------------------------------------
# Labeling Data with IndoBERT only
# -----------------------------------------

# IndoBERT classifier
bert_model = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# Salin DataFrame untuk labeling
df_bert = df.copy()

# BERT prediction
df_bert['label_bert'] = df_bert['Text'].apply(lambda x: bert_model(x)[0]['label'])

# -----------------------------------------
# Simpan hasil labeling IndoBERT
# -----------------------------------------

# Filter kolom yang relevan
df_result = df_bert[['Date', 'Text', 'Brand', 'label_bert']]

# Rename kolom untuk konsistensi
df_result.rename(columns={'label_bert': 'Sentimen'}, inplace=True)

# Simpan ke file CSV baru
df_result.to_csv('../data/output/labelled.csv', index=False)

print('Labelling data saved to ../data/output/labelled.csv')
