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
## nltk
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

file_data = '/home/atrkeffect/nltk_data'
nltk.download('all', download_dir=file_data)
nltk.data.path.clear()
nltk.data.path.append(file_data)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('indonesian'))  # Using Indonesian stopwords

def normalize_text_nltk(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Reconstruct text
    return ' '.join(words)

# Apply normalization to 'Text' column
df['Normalized_Text_NLTK'] = df['Text'].astype(str).apply(normalize_text_nltk)

## Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def normalize_text_sastrawi(text):
    text = text.lower()
    text = stemmer.stem(text)  # Perform stemming
    return text

df["Normalized_Text_Sastrawi"] = df["Text"].astype(str).apply(normalize_text_sastrawi)

## Spacy
import spacy

nlp = spacy.load("en_core_web_sm")

def normalize_text_spacy(text):
    text = text.lower()  # Lowercase
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatize & remove stopwords
    return " ".join(words)

df["Normalized_Text_Spacy"] = df["Text"].astype(str).apply(normalize_text_spacy)

## TextBlob
from textblob import TextBlob

def normalize_text_textblob(text):
    text = text.lower()
    words = TextBlob(text).words.singularize()  # Lemmatization
    return " ".join(words)

df["Normalized_Text_TextBlob"] = df["Text"].astype(str).apply(normalize_text_textblob)

#Elimination
df_normalization = df[['Date','Text', 'Normalized_Text_Spacy']]
#save normalization
df_normalization.to_csv('../data/output/normalization.csv', index=False)

#---------------------------------------------------------------------------------------
# Tekonize the texts
#---------------------------------------------------------------------------------------

# tokenize
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha]
    return tokens

# Tokenize the texts
df_normalization['Tokens'] = df_normalization['Normalized_Text_Spacy'].astype(str).apply(tokenize_text)
df_tokenization = df_normalization[['Date','Text','Normalized_Text_Spacy', 'Tokens']]

#---------------------------------------------------------------------------------------
# Stopword Removal
#---------------------------------------------------------------------------------------

from spacy.lang.id.stop_words import STOP_WORDS  # Indonesian stopwords

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in STOP_WORDS]

df_tokenization['Tokens_no_stopword'] = df_tokenization['Tokens'].apply(remove_stopwords)
df_stopword = df_tokenization[['Date','Text', 'Normalized_Text_Spacy', 'Tokens', 'Tokens_no_stopword']]


#---------------------------------------------------------------------------------------
# save seperate file
#---------------------------------------------------------------------------------------

# save tokens
df_token = df_stopword[['Date','Text', 'Tokens']]
df_token.to_csv('../data/output/tokens.csv', index=False)

#save token no stopwords
df_token_no_stopword = df_stopword[['Date','Text', 'Tokens_no_stopword']]
df_token_no_stopword.to_csv('../data/output/tokens_no_stopword.csv', index=False)