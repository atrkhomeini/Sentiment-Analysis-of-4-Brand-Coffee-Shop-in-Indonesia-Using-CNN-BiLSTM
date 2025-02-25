import os
import re
import mysql.connector
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import spacy
from textblob import TextBlob
from spacy.lang.id.stop_words import STOP_WORDS  # Indonesian stopwords


class TextPreprocessing:
    def __init__(self, db_config):
        """
        Initialize connection to MySQL database and load NLP tools.
        """
        self.db_config = db_config
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize Sastrawi
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Initialize NLTK with only necessary downloads
        nltk.download('all')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('indonesian'))  # Indonesian stopwords

        # Load data
        self.df = self.fetch_data()

    def fetch_data(self):
        """
        Connect to MySQL database and fetch text data.
        """
        try:
            connect = mysql.connector.connect(**self.db_config)
            cursor = connect.cursor()

            query = "SELECT Text FROM kopi"
            cursor.execute(query)

            # Load data into DataFrame
            df = pd.DataFrame(cursor.fetchall(), columns=['Text'])
            cursor.close()
            connect.close()
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def normalize_text_nltk(self, text):
        """
        Normalize text using NLTK: Lowercasing, removing special chars, stopwords, and lemmatization.
        """
        text = text.lower()
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        words = word_tokenize(text)  # Tokenization
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]  # Lemmatization
        return " ".join(words)

    def normalize_text_sastrawi(self, text):
        """
        Normalize text using Sastrawi: Lowercasing & stemming for Indonesian text.
        """
        text = text.lower()
        return self.stemmer.stem(text)

    def normalize_text_spacy(self, text):
        """
        Normalize text using SpaCy: Lemmatization & stopword removal.
        """
        text = text.lower()
        doc = self.nlp(text)
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatize & Remove Stopwords
        return " ".join(words)

    def normalize_text_textblob(self, text):
        """
        Normalize text using TextBlob: Lemmatization.
        """
        text = text.lower()
        words = TextBlob(text).words.singularize()  # Lemmatization
        return " ".join(words)

    def apply_normalization(self):
        """
        Apply all normalization methods and store results in DataFrame.
        """
        self.df["Normalized_Text_NLTK"] = self.df["Text"].astype(str).apply(self.normalize_text_nltk)
        self.df["Normalized_Text_Sastrawi"] = self.df["Text"].astype(str).apply(self.normalize_text_sastrawi)
        self.df["Normalized_Text_Spacy"] = self.df["Text"].astype(str).apply(self.normalize_text_spacy)
        self.df["Normalized_Text_TextBlob"] = self.df["Text"].astype(str).apply(self.normalize_text_textblob)

    def tokenize_text(self, text):
        """
        Tokenization using SpaCy.
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc if token.is_alpha]
        return tokens

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokenized text.
        """
        return [word for word in tokens if word.lower() not in STOP_WORDS]

    def apply_tokenization(self):
        """
        Apply tokenization and stopword removal.
        """
        self.df["Tokens"] = self.df["Normalized_Text_Spacy"].astype(str).apply(self.tokenize_text)
        self.df["Tokens_No_Stopwords"] = self.df["Tokens"].apply(self.remove_stopwords)

    def get_unique_filename(self, base_path):
        """
        Generate a unique filename if the file already exists.
        Example:
            - If "tokens.csv" exists, save as "tokens_1.csv"
            - If "tokens_1.csv" exists, save as "tokens_2.csv", and so on.
        """
        if not os.path.exists(base_path):
            return base_path  # If no file exists, return original filename

        filename, ext = os.path.splitext(base_path)
        counter = 1

        while os.path.exists(f"{filename}_{counter}{ext}"):
            counter += 1

        return f"{filename}_{counter}{ext}"

    def save_results(self):
        """
        Save processed data to CSV files with unique filenames if necessary.
        """
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

        # Generate unique filenames
        token_file = self.get_unique_filename(os.path.join(output_dir, "tokens.csv"))
        token_no_stopword_file = self.get_unique_filename(os.path.join(output_dir, "tokens_no_stopword.csv"))

        # Save tokens
        df_token = self.df[['Text', 'Tokens']]
        df_token.to_csv(token_file, index=False)
        print(f"Tokens saved as: {token_file}")

        # Save tokens without stopwords
        df_token_no_stopword = self.df[['Text', 'Tokens_No_Stopwords']]
        df_token_no_stopword.to_csv(token_no_stopword_file, index=False)
        print(f"Tokens without stopwords saved as: {token_no_stopword_file}")

    def run_pipeline(self):
        """
        Execute full text preprocessing pipeline.
        """
        print("Fetching data from MySQL...")
        self.apply_normalization()
        print("Applying normalization...")
        self.apply_tokenization()
        print("Tokenizing and removing stopwords...")
        self.save_results()
        print("Pipeline completed successfully!")


# --------------------------------------------------------------------------------------
# Running the Class
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # MySQL Connection Config
    db_config = {
        "host": "localhost",
        "user": "axel",
        "password": "Time12:30",
        "database": "tugas_akhir"
    }

    # Initialize and Run Preprocessing
    processor = TextPreprocessing(db_config)
    processor.run_pipeline()
