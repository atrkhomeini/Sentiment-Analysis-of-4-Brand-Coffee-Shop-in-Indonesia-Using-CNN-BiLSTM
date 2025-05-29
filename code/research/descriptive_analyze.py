#---------------------------------------------------------------------------------------
# Descriptive Analysis of Labeled Data
#---------------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('../data/output/indobert_labeled_data.csv')

# Pastikan kolom 'Brand' dan 'Sentiment' ada
assert 'Brand' in df.columns and 'Label_Bert' in df.columns

#Hitung distribusi sentimen per brand
summary = df.groupby(['Brand', 'Label_Bert']).size().unstack(fill_value=0)

# Tambahkan persentase dari total tweet per brand
summary_percent = summary.div(summary.sum(axis=1), axis=0) * 100
summary_percent = summary_percent.round(2)

import matplotlib.pyplot as plt

# Bar chart absolute count
summary.plot(kind='bar', stacked=False, colormap='viridis')
plt.title("Distribusi Sentimen per Brand")
plt.xlabel("Brand")
plt.ylabel("Jumlah Tweet")
plt.legend(title="Sentiment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Bar chart percentage
summary_percent.plot(kind='bar', stacked=False, colormap='plasma')
plt.title("Persentase Sentimen per Brand")
plt.xlabel("Brand")
plt.ylabel("Persentase")
plt.legend(title="Sentiment", loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

