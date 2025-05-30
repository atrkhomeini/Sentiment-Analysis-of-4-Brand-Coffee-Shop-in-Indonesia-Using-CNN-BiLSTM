import pandas as pd 

df = pd.read_csv('../data/output/normalized_coffee_shop_data.csv')

#---------------------------------------------------------------------------------------
# Filter tweet berisi hanya brand tanpa opini
#---------------------------------------------------------------------------------------
brands = ['fore', 'kopi kenangan', 'point coffee', 'tomoro']

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
# Save the expanded dataset
path_save = '../data/output'
df_expanded.to_csv(f'{path_save}/expanded_coffee_shop_data.csv', index=False)
print("Data expansion completed successfully!")
