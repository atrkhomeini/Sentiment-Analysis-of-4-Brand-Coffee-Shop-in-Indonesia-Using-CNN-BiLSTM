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
query = "SELECT Date, Text FROM kopi"
cursor.execute(query)

# Load data into DataFrame
df = pd.DataFrame(cursor.fetchall(), columns=['Date','Text'])
cursor.close()
connect.close()

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  

# Format the date to "mm/yyyy"
df['Date'] = df['Date'].dt.strftime('%m/%Y')


#---------------------------------------------------------------------------------------
# Normalize the texts
#---------------------------------------------------------------------------------------
import re
import json
import pandas as pd

# Load slang dictionary
with open ('../assets/NLP_bahasa_resources/combined_slang_words.txt','r', encoding="utf-8") as file:
    slang_dict = json.load(file)

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
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text  # Make sure to return the cleaned text!

# Function to replace slang words
def normalize_slang(text):
    if pd.isna(text) or text.strip() == "":  # Prevent NoneType errors
        return ""

    words = text.split()  # Tokenization using spaces
    normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
    return ' '.join(normalized_words)

# Apply normalization
df['Normalized_Text_NLTK'] = df['Text'].astype(str).apply(normalize_text)
df['Normalized_Text_Slang'] = df['Normalized_Text_NLTK'].apply(normalize_slang)

# Save the final normalized dataset
df.to_csv("tokens_normalized.csv", index=False)

print("Normalization completed successfully! Data saved as tokens_normalized.csv")
