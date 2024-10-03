import pandas as pd

# load the data
df = pd.read_csv('data/fore.csv')

# handle missing values
drop = ['image_url', 'in_reply_to_screen_name']
df.drop(drop, axis=1, inplace=True)
## drop rows with missing values
df.dropna(inplace=True)

# preprocess full_text
## remove URLs
df['full_text'] = df['full_text'].str.replace('http\S+|www.\S+', '', case=False)
## remove special characters
df['full_text'] = df['full_text'].str.replace('[^A-Za-z0-9]+', ' ')
## remove emojis
df['full_text'] = df['full_text'].str.encode('ascii', 'ignore').str.decode('ascii')
## remove mentions
df['full_text'] = df['full_text'].str.replace('@\S+', '', case=False)

# save the cleaned data
df.to_csv('data/fore_cleaned.csv', index=False)