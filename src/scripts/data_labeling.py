import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../data/output/expanded_coffee_shop_data.csv')

# -----------------------------------------
# Labeling Data with IndoBERT only
# -----------------------------------------

# IndoBERT classifier
bert_model = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# Salin DataFrame untuk labeling
df_bert = df.copy()

# BERT prediction
df_bert['label_bert'] = df_bert['Text'].apply(lambda x: bert_model(x)[0]['label'])

# Plotting the results
df_bert['label_bert'].value_counts().plot(kind='bar')
plt.title("Distribusi Label IndoBERT")
plt.ylabel("Jumlah Tweet")
plt.xlabel("Sentimen")
plt.xticks(rotation=0)
plt.show()

# -----------------------------------------
# Simpan hasil labeling IndoBERT
# -----------------------------------------

# Filter kolom yang relevan
df_result = df_bert[['Date', 'Text', 'Brand', 'label_bert']]

# Rename kolom untuk konsistensi
df_result.rename(columns={'label_bert': 'Sentimen'}, inplace=True)

# Simpan ke file CSV baru
df_result.to_csv('../data/output/labelled.csv', index=False)

print("Filtered data saved to '../data/output/test_indobert_labeled_data.csv'")
