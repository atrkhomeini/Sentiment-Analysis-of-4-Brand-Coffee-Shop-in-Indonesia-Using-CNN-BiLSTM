import pandas as pd 

df = pd.read_csv('../data/output/normalized_coffee_shop_data.csv')

#---------------------------------------------------------------------------------------
# Filter tweet yang menyebut Fore dan bukan hanya daftar brand
# ---------------------------------------------------------------------------------------

# Fokus hanya ke brand Fore
target_brand = 'fore'
brands = ['fore', 'kopi kenangan', 'point coffee', 'tomoro', 'kopken']

def detect_brands(text):
    return [brand for brand in brands if brand in text.lower()]

def is_brand_only(text):
    words = text.lower().split()
    return all(word in brands for word in words)

# Filter hanya baris yang menyebut Fore dan bukan hanya kumpulan brand
filtered_rows = []
for _, row in df.iterrows():
    text = row['Text Normalization']
    if target_brand in text.lower() and not is_brand_only(text):
        filtered_rows.append({
            'Date': row['Date'],
            'Text': text,
            'Brand': target_brand,
        })

# Buat DataFrame hasil
df_expanded = pd.DataFrame(filtered_rows)

# Simpan hasil ekspansi
path_save = '../data/output'
df_expanded.to_csv(f'{path_save}/expanded_fore_data.csv', index=False)
print("Data expansion (Fore only) completed successfully!")
#---------------------------------------------------------------------------------------
# Descriptive Analysis of Brand Mentions in Tweets
#---------------------------------------------------------------------------------------
# Hitung jumlah brand per tweet
df["brand_count"] = df["Text Normalization"].apply(lambda x: len(detect_brands(x)))

# 1. Tweet dengan 1 brand saja
single_brand_tweets = df[df["brand_count"] == 1]

# 2. Tweet dengan lebih dari 1 brand
multi_brand_tweets = df[df["brand_count"] > 1]

# 3. Hitung berapa baris hasil split yang berasal dari tweet multi-brand
multi_brand_texts = set(multi_brand_tweets["Text Normalization"])
split_from_multi = df_expanded[df_expanded["Text"].isin(multi_brand_texts)]

# Tampilkan hasil
print("Jumlah tweet dengan 1 brand:", len(single_brand_tweets))
print("Jumlah tweet dengan multiple brand:", len(multi_brand_tweets))
print("Jumlah baris hasil split dari tweet multiple brand:", len(split_from_multi))

df_expanded['Brand'].value_counts()