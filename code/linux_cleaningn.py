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
query = "SELECT text FROM kopi"
cursor.execute(query)

# Load data into DataFrame
df = pd.DataFrame(cursor.fetchall(), columns=['text'])
cursor.close()
connect.close()

#---------------------------------------------------------------------------------------
# Normalize the texts
#---------------------------------------------------------------------------------------
import os
import nltk
from nltk.tokenize import word_tokenize

file_data = '/home/atrkeffect/nltk_data'
nltk.download('all', download_dir=file_data)
nltk.data.path.clear()
nltk.data.path.append(file_data)

# Convert texts to lowercase
df['texts'] = df['texts'].str.lower()


#---------------------------------------------------------------------------------------
# Tekonize the texts
#---------------------------------------------------------------------------------------

# tokenize



# Tokenize the texts
df['tokens'] = df['texts'].apply(word_tokenize)