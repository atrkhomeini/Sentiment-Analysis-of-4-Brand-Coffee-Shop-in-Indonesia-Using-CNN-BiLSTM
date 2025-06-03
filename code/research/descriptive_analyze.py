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

# Hitung distribusi sentimen untuk setiap brand
brand_distribution = df.groupby(['Brand', 'Label_Bert']).size().reset_index(name='Count')

# Tampilkan hasil distribusi per brand dalam bentuk DataFrame
print("Distribusi Sentimen untuk Setiap Brand:")
print(brand_distribution)

#----------------------------------------------------------------------
# world cloud
#----------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Daftar kata brand yang ingin dihapus
brands_to_remove = ['fore', 'kopi kenangan', 'point coffee', 'tomoro', 'kopi', 'kenangan', 'coffee', 'tuku', 'janjiw', 'point']

def remove_brand_words(text):
    text = text.lower()
    for brand in brands_to_remove:
        text = text.replace(brand.lower(), '')
    return text.strip()

def generate_wordcloud(texts, title):
    all_words = ' '.join(texts).lower().split()
    word_freq = Counter(all_words)
    
    if not word_freq:
        print(f"No words to display for: {title}")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 🔹 Word Cloud per Sentimen (tanpa brand)
for sentiment in df['Label_Bert'].unique():
    subset = df[df['Label_Bert'] == sentiment].copy()
    subset['Cleaned_Text'] = subset['Text'].apply(remove_brand_words)
    generate_wordcloud(subset['Cleaned_Text'], f"Word Cloud - Sentiment: {sentiment}")

# 🔹 Word Cloud per Brand dan Sentimen (tanpa brand)
brands = df['Brand'].unique()

for brand in brands:
    for sentiment in df['Label_Bert'].unique():
        subset = df[(df['Brand'] == brand) & (df['Label_Bert'] == sentiment)].copy()
        if len(subset) > 0:
            subset['Cleaned_Text'] = subset['Text'].apply(remove_brand_words)
            generate_wordcloud(subset['Cleaned_Text'], f"{brand.title()} - {sentiment.title()}")

