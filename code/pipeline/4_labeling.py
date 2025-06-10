import pandas as pd

# load the data
df = pd.read_csv('../data/output/expanded_fore_data.csv')

#-----------------------------------------
# Labeling Data with BERT and VADER
#-----------------------------------------
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# IndoBERT classifier
bert_model = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# VADER
vader = SentimentIntensityAnalyzer()

# Salin DataFrame untuk eksperimen
df_compare = df.copy()

# BERT prediction
df_compare['label_bert'] = df_compare['Text'].apply(lambda x: bert_model(x)[0]['label'])

# VADER prediction
def vader_label(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df_compare['label_vader'] = df_compare['Text'].apply(vader_label)

# Bandingkan
df_compare['agreement'] = df_compare['label_bert'] == df_compare['label_vader']
agreement_ratio = df_compare['agreement'].mean()
print(f"Agreement ratio between BERT and VADER: {agreement_ratio:.2%}")

#Plotting the results
import matplotlib.pyplot as plt

df_compare[['label_bert', 'label_vader']].apply(pd.Series.value_counts).plot(kind='bar')
plt.title("Perbandingan Distribusi Label IndoBERT vs VADER")
plt.ylabel("Jumlah Tweet")
plt.xlabel("Sentimen")
plt.xticks(rotation=0)
plt.show()

#--------------------------------------------------------------------------------------------------
# IndoBERT choose the most confident label
#--------------------------------------------------------------------------------------------------

# Filter the DataFrame to include only relevant columns
df_result = df_compare[['Date', 'Text', 'Brand', 'label_bert']]

# Rename the column for consistency
df_result.rename(columns={'label_bert': 'Label_Bert'}, inplace=True)

# Save the result to a new CSV file
df_result.to_csv('../data/output/indobert_labeled_data.csv', index=False)

print("Filtered data saved to '../data/output/test_indobert_labeled_data.csv'")

