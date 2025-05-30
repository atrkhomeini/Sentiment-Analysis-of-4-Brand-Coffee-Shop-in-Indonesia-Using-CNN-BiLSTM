#---------------------------------------------------------------------------------------
# Connect to MySQL database and fetch data
#---------------------------------------------------------------------------------------

import mysql.connector
import pandas as pd

connect = mysql.connector.connect(
    host = 'localhost',
    user = 'axel',
    password = 'Time12:30',
    database = 'tugas_akhir'
)

cursor = connect.cursor()

#fetch data
query = "SELECT dates, texts FROM coffee_shop ORDER BY dates"
cursor.execute(query)

# Load data into DataFrame
df = pd.DataFrame(cursor.fetchall(), columns=['dates','texts'])
cursor.close()
connect.close()

# Convert Date column to datetime format
df['dates'] = pd.to_datetime(df['dates'], errors='coerce')  

# Format the date to "mm/yyyy"
df['dates'] = df['dates'].dt.strftime('%m/%Y')

#----------------------------------------------------------------------------------------
# Consistency, Missing Values, and Duplicates
#----------------------------------------------------------------------------------------
# Rename columns for consistency
df.rename(columns={'dates': 'Date', 'texts': 'Text'}, inplace=True)
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)
# Check for duplicate texts
duplicate_texts = df['Text'].duplicated().sum()
print("Duplicate Texts:", duplicate_texts)
# Remove rows with missing or duplicate texts
df.dropna(subset=['Text'], inplace=True)
df.drop_duplicates(subset=['Text'], inplace=True)
# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

#---------------------------------------------------------------------------------------
# Normalize the texts using both slang and root word dictionaries
#---------------------------------------------------------------------------------------
import re
import json
import pandas as pd

# Load slang dictionary
try:
    with open('../src/NLP_bahasa_resources/combined_slang_words.txt', 'r', encoding="utf-8") as slang_file:
        slang_dict = json.load(slang_file)
except FileNotFoundError:
    print("Error: Slang dictionary file not found. Please check the file path.")
    slang_dict = {}
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from the slang dictionary file.")
    slang_dict = {}

# Load stop words dictionary
try:
    with open('../assets/NLP_bahasa_resources/combined_stop_words.txt', 'r', encoding="utf-8") as stop_words_file:
        stop_words = set(stop_words_file.read().splitlines())
except FileNotFoundError:
    print("Error: Stop words dictionary file not found. Please check the file path.")
    stop_words = set()

# Function to clean text
def normalize_text(text):
    if pd.isna(text):  # Ensure no None values
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and punctuation but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text  # Make sure to return the cleaned text!

# Function to replace slang words only
def normalize_slang(text):
    if pd.isna(text) or text.strip() == "":  # Prevent NoneType errors
        return ""

    words = text.split()  # Tokenization using spaces
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Function to remove stop words
def remove_stop_words(text):
    if pd.isna(text) or text.strip() == "":  # Prevent NoneType errors
        return ""

    words = text.split()  # Tokenization using spaces
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply normalization
df['Text Normalization'] = df['Text'].astype(str).apply(normalize_text).apply(normalize_slang).apply(remove_stop_words)

print("Data normalization completed successfully!")
# Save the final normalized dataset
path_save = '../data/output'
df.to_csv(f'{path_save}/normalized_coffee_shop_data.csv', index=False)
#---------------------------------------------------------------------------------------
