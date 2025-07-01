import yaml
import pandas as pd
import google.generativeai as genai
import time
import json
import os

# Load API key from the YAML file
with open('../key/key.yml', 'r') as file:
    keys = yaml.safe_load(file)
    GEMINI_API_KEY = keys.get('GEMINI_API_KEY')

print("API key loaded successfully.")

# Ensure the client and model are correctly initialized
client = genai.configure(api_key=GEMINI_API_KEY)

# Inisialisasi model Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

df = pd.read_excel('../data/data_valid/main_indobert_train.xlsx', sheet_name='data_test')
# Siapkan kolom hasil
if 'Predicted_Aspect' not in df.columns:
    df['Predicted_Aspect'] = ""
# Checkpoint file untuk progress
checkpoint_file = 'progress_checkpoint.json'


# Load progress jika ada
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        progress = json.load(f)
    print("Checkpoint loaded.")
else:
    progress = {}

# Looping untuk Zero-Shot Classification
save_interval = 10  # Simpan setiap 10 tweet
counter = 0


for idx, row in df.iterrows():
    tweet = row['Text']
    
    prompt = f"""
    Tweet: "{tweet}"
    Available Aspects: Produk, Layanan, Promosi.
    Question: Which aspect does the tweet talk about? Only answer one aspect: Produk, Layanan, or Promosi. Do not explain.
    """
    
    try:
        response = model.generate_content([prompt])
        aspect = response.text.strip()

        df.at[idx, 'Predicted_Aspect'] = aspect
        progress[str(idx)] = aspect  # Simpan progress
        counter += 1

        print(f"Processed tweet {idx+1}/{len(df)}: Aspect = {aspect}")

        # Simpan setiap N tweet
        if counter % save_interval == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(progress, f)
            df.to_excel('tweet_with_predicted_aspect_test.xlsx', index=False)
            print(f"Progress saved after {counter} tweets.")

        time.sleep(2.0)  # Hindari rate-limit

    except Exception as e:
        print(f"Error processing tweet {idx+1}: {e}")
        df.at[idx, 'Predicted_Aspect'] = 'ERROR'
        continue

# Simpan file akhir
with open(checkpoint_file, 'w') as f:
    json.dump(progress, f)
df.to_excel('tweet_with_predicted_aspect_test.xlsx', index=False)
print("Aspect prediction completed and final progress saved.")