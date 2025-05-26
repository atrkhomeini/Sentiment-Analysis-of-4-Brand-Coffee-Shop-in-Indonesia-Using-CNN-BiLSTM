#--------------------------------------------------------
# Script untuk memeriksa struktur kolom dari file CSV yang dihasilkan oleh crawler
#--------------------------------------------------------
import os
import pandas as pd

# Folder tempat file CSV berada
folder_path = '../data/output/crawl'

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

column_structures = {}

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        # Pakai sep=';' untuk file dengan pemisah titik koma
        df = pd.read_csv(file_path, sep=';', nrows=5)
        columns = tuple(df.columns)
        column_structures[file] = columns
    except Exception as e:
        column_structures[file] = f"Error: {e}"

reference_structure = list(column_structures.values())[0]

print("=== HASIL PENGECEKAN STRUKTUR KOLOM CSV ===")
for file, structure in column_structures.items():
    if isinstance(structure, str):
        print(f"{file}: GAGAL DIBACA ({structure})")
    elif structure == reference_structure:
        print(f"{file}: ✅ Struktur kolom SAMA")
    else:
        print(f"{file}: ❌ Struktur kolom BERBEDA")
        print(f"    -> {structure}")

print("\nStruktur kolom referensi (dari file pertama):")
print(reference_structure)

#--------------------------------------------------------
# Script untuk merge
#---------------------------------------------------------

# Inisialisasi list untuk menampung DataFrame
df_list = []

# Membaca dan menambahkan setiap CSV ke list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep=';')  # Gunakan sep=';' sesuai dengan format file
    df['source_file'] = file  # Optional: tandai asal file
    df_list.append(df)

# Gabungkan semua DataFrame menjadi satu
merged_df = pd.concat(df_list, ignore_index=True)

# Simpan ke file baru
output_file = 'merged_crawling_result.csv'
merged_df.to_csv(output_file, index=False)

print(f"Merge selesai! File disimpan sebagai: {output_file}")