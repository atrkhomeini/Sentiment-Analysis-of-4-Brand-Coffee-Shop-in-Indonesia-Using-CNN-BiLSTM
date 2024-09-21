import tweepy
import pandas as pd
from key import consumer_key, consumer_secret, access_key, access_secret

# Twitter API credentials
consumer_key = "CONSUMER_KEY"
consumer_secret = "CONSUMER_SECRET"
access_key = "ACCESS_KEY"
access_secret = "ACCESS_SECRET"

#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, access_key, access_secret)

#instantiating tweepy
api = tweepy.API(auth, wait_on_rate_limit=True)

search_query = "dove -filter:retweets"
tweet_counts = 100

# gathering tweets
tweets = tweepy.Cursor(api.search, q=search_query, lang="id", tweet_mode='extended').items(tweet_counts)

# save tweets to a list
tweet_data = []
for tweet in tweets:
    tweet_data.append({
        "created_at" : tweet.created_at,
        "username" : tweet.user.screen_name,
        "text" : tweet.full_text,
        "retweet_count" : tweet.retweet_count,
        "favorite_count" : tweet.favorite_count,
        "location" : tweet.user.location
    })

print("Total tweets fetched:", len(tweet_data))

# convert list to dataframe
tweet_df = pd.DataFrame(tweet_data)

# save dataframe to csv
tweet_df.to_csv("tweets.csv", index=False, encoding='utf-8')
print("Data saved to tweets.csv")