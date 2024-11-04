import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Create fake Twitter data
def generate_fake_tweets(n=100):
    topics = ['politics', 'sports', 'technology', 'entertainment']
    tweets = []
    
    for _ in range(n):
        topic = np.random.choice(topics)
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            tweet = f"I love {topic}! It's amazing!"
        elif sentiment == 'negative':
            tweet = f"I hate {topic}. It's terrible."
        else:
            tweet = f"{topic} is neither good nor bad."
        
        tweets.append({'text': tweet, 'topic': topic})
    
    return pd.DataFrame(tweets)

# Step 2: Perform sentiment analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Generate fake tweets
df = generate_fake_tweets(200)

# Analyze sentiment
df['sentiment'] = df['text'].apply(analyze_sentiment)

# Step 3: Save results to CSV
df.to_csv('twitter_sentiment.csv', index=False)

# Step 4: Visualize the results
plt.figure(figsize=(10, 6))
df.boxplot(column='sentiment', by='topic')
plt.title('Sentiment Analysis of Fake Tweets by Topic')
plt.suptitle('')
plt.ylabel('Sentiment Polarity')
plt.savefig('sentiment_analysis.png')
plt.close()

# Display average sentiment by topic
print(df.groupby('topic')['sentiment'].mean())
    


