import pandas as pd
import matplotlib.pyplot as plt

df_sentiment = pd.read_csv('../research/df_sentiment.csv')

# Extract mentions of each coffee brand in the dataset
brands = ["tomoro", "point coffee", "fore", "kopi kenangan"]

# Create a dataframe to store sentiment distribution for each brand
brand_sentiment_counts = {}

for brand in brands:
    brand_df = df_sentiment[df_sentiment["Normalized_Text_Slang"].str.contains(brand, case=False, na=False)]
    sentiment_distribution = brand_df["Sentiment"].value_counts(normalize=True) * 100
    brand_sentiment_counts[brand] = sentiment_distribution

# Convert dictionary to DataFrame
brand_sentiment_df = pd.DataFrame(brand_sentiment_counts).fillna(0)  # Fill NaN with 0 for missing sentiment categories

# Plot sentiment distribution for each brand
brand_sentiment_df.T.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
plt.xlabel("Coffee Brand")
plt.ylabel("Percentage (%)")
plt.title("Sentiment Distribution for Each Coffee Brand")
plt.xticks(rotation=30)
plt.legend(title="Sentiment")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the visualization
plt.show()


# Initialize new columns with 0
for brand in brands:
    for sentiment in ["positive", "negative", "neutral"]:
        df_sentiment[f"{brand.replace(' ', '_')}_{sentiment}"] = 0

# Function to update sentiment columns based on the brand presence
def update_sentiment_counts(row):
    text = str(row["Normalized_Text_Slang"]).lower()  # Ensure text is in lowercase
    sentiment = row["Sentiment"]

    for brand in brands:
        if brand in text:
            # Update the corresponding sentiment column with 1
            df_sentiment.at[row.name, f"{brand.replace(' ', '_')}_{sentiment}"] = 1

# Apply function to update sentiment counts
df_sentiment.apply(update_sentiment_counts, axis=1)

# Reshape the brand sentiment summary into a DataFrame
brand_sentiment_df = df_sentiment.reset_index()
brand_sentiment_df.columns = ["Brand_Sentiment", "Count"]

# Split brand and sentiment for better visualization
brand_sentiment_df["Brand"] = brand_sentiment_df["Brand_Sentiment"].apply(lambda x: "_".join(x.split("_")[:-1]).replace("_", " "))
brand_sentiment_df["Sentiment"] = brand_sentiment_df["Brand_Sentiment"].apply(lambda x: x.split("_")[-1])

# Pivot the DataFrame for plotting
pivot_df = brand_sentiment_df.pivot(index="Brand", columns="Sentiment", values="Count")

# Plot the sentiment distribution for each brand
pivot_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
plt.xlabel("Coffee Brand")
plt.ylabel("Count")
plt.title("Sentiment Distribution for Each Coffee Brand")
plt.xticks(rotation=30)
plt.legend(title="Sentiment")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the visualization
plt.show()
