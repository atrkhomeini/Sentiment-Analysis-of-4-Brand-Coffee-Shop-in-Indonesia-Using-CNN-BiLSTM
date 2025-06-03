import pandas as pd 

df = pd.read_csv('../data/output/normalized_coffee_shop_data.csv')

#---------------------------------------------------------------------------------------
# Filter tweet berisi hanya brand tanpa opini
#---------------------------------------------------------------------------------------
brands = ['fore', 'kopi kenangan', 'point coffee', 'tomoro', 'kopken']

def detect_brands(text):
    return [brand for brand in brands if brand in text.lower()]

def is_brand_only(text):
    words = text.lower().split()
    return all(word in brands for word in words)

def split_by_brand(row):
    text = row['Text Normalization']
    detected = detect_brands(text)

    # Skip: jika tidak ada brand atau isinya hanya brand saja
    if len(detected) == 0 or is_brand_only(text):
        return None

    return pd.DataFrame({
        'Text': [text] * len(detected),
        'Brand': detected,
        'Date': [row['Date']] * len(detected)  # Include the Date column
    })

# Eksekusi pada semua baris
results = []
for _, row in df.iterrows():
    split_df = split_by_brand(row)
    if split_df is not None:
        results.append(split_df)

df_expanded = pd.concat(results, ignore_index=True)
# Ganti semua "kopken" jadi "kopi kenangan"
df_expanded['Brand'] = df_expanded['Brand'].replace('kopken', 'kopi kenangan')
 
# Save the expanded dataset
path_save = '../data/output'
df_expanded.to_csv(f'{path_save}/expanded_coffee_shop_data.csv', index=False)
print("Data expansion completed successfully!")


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