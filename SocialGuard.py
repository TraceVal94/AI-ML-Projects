import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import random 

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Function to generate fake social media posts
def generate_fake_posts(count=100):
    topics = ['customer service', 'product feedback', 'employee conduct', 'company policy']
    sentiments = ['positive', 'negative', 'neutral']
    posts = []
    
    for _ in range(count):
        topic = random.choice(topics)
        sentiment = random.choice(sentiments)
        
        if sentiment == 'positive':
            post = f"I love how the company handles {topic}! Great job!"
        elif sentiment == 'negative':
            post = f"The company's approach to {topic} is terrible. Need improvement."
        else:
            post = f"The company's {topic} seems average. Nothing special."
        
        posts.append({'text': post, 'topic': topic})
    
    return pd.DataFrame(posts)

# Sentiment Analysis
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    if score['compound'] > 0.05:
        return 'Positive'
    elif score['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Topic Classification
class TopicClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)

# Streamlit App
def main():
    st.title("Social Media Policy Compliance Analyzer")

    # Generate fake data
    df = generate_fake_posts(500)
    
    # Perform sentiment analysis
    df['sentiment'] = df['text'].apply(analyze_sentiment)
    
    # Train topic classifier
    topic_classifier = TopicClassifier()
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['topic'], test_size=0.2, random_state=42)
    topic_classifier.train(X_train, y_train)
    
    # Predict topics for all posts
    df['predicted_topic'] = topic_classifier.predict(df['text'])

    # Display results
    st.subheader("Sentiment Analysis")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Topic Distribution")
    topic_counts = df['predicted_topic'].value_counts()
    st.bar_chart(topic_counts)

    # Word Cloud
    st.subheader("Word Cloud")
    text = " ".join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Display sample posts
    st.subheader("Sample Posts")
    sample_df = df.sample(5)
    for _, row in sample_df.iterrows():
        st.write(f"Sentiment: {row['sentiment']}, Topic: {row['predicted_topic']}")
        st.write(row['text'])
        st.write("---")

    # Topic Classification Performance
    st.subheader("Topic Classification Performance")
    y_pred = topic_classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

if __name__ == "__main__":
    main()
