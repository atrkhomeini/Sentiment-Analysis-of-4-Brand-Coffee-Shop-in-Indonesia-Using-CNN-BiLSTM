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
# Normalize the texts
#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
# Tekonize the texts
#---------------------------------------------------------------------------------------

# tokenize
import os
import nltk
from nltk.tokenize import word_tokenize

# download the punkt package
file_data = 'C:/nltk_data'
nltk.download('all', download_dir=file_data)
nltk.data.path.clear()
nltk.data.path.append(file_data)

# Convert texts to lowercase
df['texts'] = df['texts'].str.lower()

# Tokenize the texts
df['tokens'] = df['texts'].apply(word_tokenize)