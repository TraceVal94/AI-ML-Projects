import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from wordcloud import WordCloud
import re
import unittest

# Sentiment analysis using Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Data preprocessing
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Generate fake tweets
def generate_fake_tweets(n=200):
    topics = ['politics', 'sports', 'technology', 'entertainment']
    tweets = []
    
    for _ in range(n):
        topic = np.random.choice(topics)
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            tweet = f"I love {topic}! It's amazing! #awesome"
        elif sentiment == 'negative':
            tweet = f"I hate {topic}. It's terrible. #disappointed"
        else:
            tweet = f"{topic} is neither good nor bad. #neutral"
        
        tweets.append({'text': tweet, 'topic': topic})
    
    return pd.DataFrame(tweets)

# Perform sentiment analysis
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

# Classify sentiment
def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Generate and process data
df = generate_fake_tweets(200)
df['processed_text'] = df['text'].apply(preprocess_text)
df['sentiment_score'] = df['processed_text'].apply(analyze_sentiment)
df['sentiment_category'] = df['sentiment_score'].apply(classify_sentiment)

# Save results to CSV
df.to_csv('twitter_sentiment.csv', index=False)

# Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(x='topic', y='sentiment_score', data=df)
plt.title('Sentiment Analysis of Fake Tweets by Topic')
plt.ylabel('Sentiment Score')
plt.savefig('sentiment_boxplot.png')
plt.close()

plt.figure(figsize=(10, 6))
df['sentiment_category'].value_counts().plot(kind='bar')
plt.title('Distribution of Sentiment Categories')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.close()

# Word cloud
all_words = ' '.join(df['processed_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Tweets')
plt.savefig('wordcloud.png')
plt.close()

# Print summary
print(df.groupby('topic')['sentiment_score'].mean())
print("\nSentiment Category Distribution:")
print(df['sentiment_category'].value_counts(normalize=True))

# Unit test
class TestSentimentAnalysis(unittest.TestCase):
    def test_classify_sentiment(self):
        self.assertEqual(classify_sentiment(0.1), 'Positive')
        self.assertEqual(classify_sentiment(-0.1), 'Negative')
        self.assertEqual(classify_sentiment(0), 'Neutral')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)