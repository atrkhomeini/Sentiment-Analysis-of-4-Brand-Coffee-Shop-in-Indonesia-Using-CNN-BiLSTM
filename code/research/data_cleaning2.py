import pandas as pd

df = pd.read_csv('../data/output/crawl/merged_crawling_result.csv', sep=',')
#---------------------------------------------------------------------------------------

df.isnull().sum()
df['Text'].duplicated().sum()
