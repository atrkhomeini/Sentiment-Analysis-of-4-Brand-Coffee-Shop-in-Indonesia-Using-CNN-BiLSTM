import pandas as pd

# load the data
filename='kopken.csv'
file_path = f'data/{filename}'
df = pd.read_csv(file_path)

# handle missing values
drop = ['image_url', 'in_reply_to_screen_name', 'tweet_url', 'lang', 'username']
df.drop(drop, axis=1, inplace=True)
## drop rows with missing values
df.dropna(inplace=True)

# preprocess full_text
## remove URLs
df['full_text'] = df['full_text'].str.replace('http\S+|www.\S+', '', regex=True)
## remove special characters
df['full_text'] = df['full_text'].str.replace('[^A-Za-z0-9]+', ' ')
## remove emojis
df['full_text'] = df['full_text'].str.encode('ascii', 'ignore').str.decode('ascii')
## remove mentions
df['full_text'] = df['full_text'].str.replace('@\S+', '', regex=True)
## remove hashtags
df['full_text'] = df['full_text'].str.replace('#\S+', '', regex=True)
## convert uppercase to lowercase
df['full_text'] = df['full_text'].str.lower()
## delete stopwords
#!pip install nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop = stopwords.words('indonesian')
df['full_text'] = df['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
'''## tokenize the full_text
nltk.download('punkt')
from nltk.tokenize import word_tokenize
df['tokens'] = df['full_text'].apply(word_tokenize)
'''


# save the cleaned data
filename_saved = 'kopken_cleaned.csv'
file_path_saved = f'data/{filename_saved}'
df.to_csv(file_path_saved, index=False)