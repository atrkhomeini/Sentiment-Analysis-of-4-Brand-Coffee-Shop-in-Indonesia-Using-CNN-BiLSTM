import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../../data/output/normalized.csv')

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
df_result = df_bert[['Date', 'Text','Text Normalization', 'label_bert']]

# Rename kolom untuk konsistensi
df_result.rename(columns={'label_bert': 'Sentimen'}, inplace=True)

# Simpan ke file CSV baru
df_result.to_excel('../../data/output/fore_labelled.xlsx', index=False)

print('Labelling data saved to ../../data/output/fore_labelled.xlsx')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel('../../data/use/merge/labelled_merge.xlsx', sheet_name='Sheet1')
# Visualisasi Distribusi Sentimen
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Sentimen', palette='viridis', order=df['Sentimen'].value_counts().index)
plt.title('Distribusi Sentimen Hasil Pelabelan IndoBERT')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Data')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

plt.tight_layout()
plt.show()
# Anda juga bisa mencetak jumlah pastinya:
print("\nJumlah tweet per kategori sentimen setelah pelabelan IndoBERT:")
print(df['Sentimen'].value_counts())