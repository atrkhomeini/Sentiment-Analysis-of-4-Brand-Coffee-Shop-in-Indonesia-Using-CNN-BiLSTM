#---------------------------------------------------------------------------------------
# Connect to MySQL database and fetch data
#---------------------------------------------------------------------------------------

import mysql.connector
import pandas as pd

connect = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = 'Time12:30',
    database = 'tugas_akhir'
)

cursor = connect.cursor()

#fetch data
query = "SELECT texts FROM kopi"
cursor.execute(query)

# Load data into DataFrame
df = pd.DataFrame(cursor.fetchall(), columns=['texts'])
cursor.close()
connect.close()

#---------------------------------------------------------------------------------------
# Sentiment Analysis using GloVe
#---------------------------------------------------------------------------------------

# tokenize
import os
import nltk
from nltk.tokenize import word_tokenize

# download the punkt package
nltk.download('punkt', download_dir='C:/nltk_data')
nltk.data.path.clear()
nltk.data.path.append("C:/nltk_data/tokenizers/punkt")

try:
    print(nltk.data.find('tokenizers/punkt'))
    print("Punkt tokenizer is correctly installed!")
except LookupError:
    print("Punkt tokenizer is still missing. Check installation.")

# Define the function
def nltk_tokenize(text):
    if isinstance(text, str):  # Ensure text is a string
        return word_tokenize(text.lower())  # Convert to lowercase and tokenize
    else:
        return []  # Return an empty list if the text is not valid

# Apply the function to the correct column (check column name first!)
if 'texts' in df.columns:  # Ensure the column exists
    df['tokens'] = df['texts'].apply(nltk_tokenize)
    print(df[['texts', 'tokens']].head())  # Show results
else:
    print("Error: Column 'Text' does not exist in the dataframe!")