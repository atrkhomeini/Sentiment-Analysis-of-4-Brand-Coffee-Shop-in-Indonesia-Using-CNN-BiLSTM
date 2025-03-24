#------------------------------------------------------------------------------------------------------------------
# Description: This script is used to analyze the sentiment distribution for each coffee brand in the dataset.
#------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

df_sentiment = pd.read_csv('../data/output/3_labeling.csv')

# Extract mentions of each coffee brand in the dataset
brands = ["tomoro", "point coffee", "fore", "kopi kenangan"]

# Step 1: Identify rows with multiple brand mentions
df_sentiment["Brand_Mentioned"] = df_sentiment["Normalized_Text_Slang"].apply(
    lambda text: [brand for brand in brands if brand in str(text).lower()]
)
df_sentiment["Brand_Mention_Count"] = df_sentiment["Brand_Mentioned"].apply(len)

# Option 1: Exclude tweets with 2 or more brand mentions
df_single_brand = df_sentiment[df_sentiment["Brand_Mention_Count"] == 1].copy()

# Option 2: Split tweets with multiple brand mentions into multiple rows (one per brand)
multi_brand_rows = df_sentiment[df_sentiment["Brand_Mention_Count"] > 1]

# Create new rows by repeating the tweet for each brand it mentions
split_rows = []
for _, row in multi_brand_rows.iterrows():
    for brand in row["Brand_Mentioned"]:
        new_row = row.copy()
        new_row["Normalized_Text_Slang"] = str(new_row["Normalized_Text_Slang"])
        new_row["Brand_Mentioned"] = [brand]  # now it's single
        new_row["Brand"] = brand
        split_rows.append(new_row)

df_split_multi_brand = pd.DataFrame(split_rows)

# Combine both datasets: single-brand tweets + split multi-brand tweets
df_for_analysis = pd.concat([
    df_single_brand.assign(Brand=lambda d: d["Brand_Mentioned"].str[0]),  # unwrap single brand
    df_split_multi_brand
], ignore_index=True)

#-------------------------------------------------------------------------------------------------------------------------------
# Plot sentiment distribution for each brand
#-------------------------------------------------------------------------------------------------------------------------------
import ast
# plot for Option 1
df_single_brand["Brand_Mentioned"] = df_single_brand["Brand_Mentioned"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Ambil brand pertama dari list
df_single_brand["Brand"] = df_single_brand["Brand_Mentioned"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
)

# Group by and count
summary_single = df_single_brand.groupby(["Brand", "Sentiment"]).size().unstack(fill_value=0)

# Plot
summary_single.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
plt.title("Sentiment Distribution (Single Brand Only)")
plt.xlabel("Brand")
plt.ylabel("Tweet Count")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# plot for Option 2
# Hitung jumlah sentimen per brand
summary_split = df_split_multi_brand.groupby(["Brand", "Sentiment"]).size().unstack(fill_value=0)

# Plot
summary_split.plot(kind="bar", figsize=(10, 6), colormap="plasma", edgecolor="black")
plt.title("Sentiment Distribution (Multi-Brand Tweets Split per Row)")
plt.xlabel("Brand")
plt.ylabel("Tweet Count")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()