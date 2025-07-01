import pandas as pd 

# Load dataset
df = pd.read_csv('../data/output/normalized.csv')

# ---------------------------------------------------------------------------------------
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
            'Text': row['Text'],  # Tambahkan kolom Text asli
            'Text Normalization': text,
            'Brand': target_brand,
        })

# Buat DataFrame hasil
df_expanded = pd.DataFrame(filtered_rows)

# Simpan hasil ekspansi
path_save = '../data/output'
df_expanded.to_excel(f'{path_save}/elimination.xlsx', index=False)
print('Elimination data saved to ../data/output/elimination.csv')
