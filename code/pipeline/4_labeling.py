import pandas as pd

# load the data
df = pd.read_csv('../data/output/normalized_coffee_shop_data.csv')

#-----------------------------------------
# Labeling Data with BERT and VADER
#-----------------------------------------
from transformers import pipeline

# IndoBERT classifier
bert_model = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# Salin DataFrame untuk eksperimen
df_compare = df.copy()

# BERT prediction
df_compare['label_bert'] = df_compare['Text'].apply(lambda x: bert_model(x)[0]['label'])

#Plotting the results
import matplotlib.pyplot as plt

df['Label_Bert'].apply(pd.Series.value_counts).plot(kind='bar')
plt.title("Perbandingan Distribusi Label IndoBERT vs VADER")
plt.ylabel("Jumlah Tweet")
plt.xlabel("Sentimen")
plt.xticks(rotation=0)
plt.show()

#--------------------------------------------------------------------------------------------------
# IndoBERT choose the most confident label
#--------------------------------------------------------------------------------------------------

# Rename the column for consistency
df_compare.rename(columns={'label_bert': 'Label_Bert'}, inplace=True)

# Save the result to a new CSV file
df_compare.to_csv('../data/output/indobert_labeled_data.csv', index=False)

print("Filtered data saved to '../data/output/test_indobert_labeled_data.csv'")

