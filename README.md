                                                        ### WELCOME TO MY AI/ML PORTFOLIO ###

## PROJECT 1 - SENTIMENT ANALYSIS TOOL ##

This project is a basic sentiment analysis tool that analyzes fake Twitter data. It demonstrates fundamental skills in Python, natural language processing (NLP), and data visualization.

#Features

- Generates fake Twitter data
- Performs basic sentiment analysis using TextBlob
- Saves results to a CSV file
- Visualizes results using matplotlib

#requirements

- Python 3.x
- pandas
- numpy
- textblob
- matplotlib

#Usage

1. Ensure you have all required libraries installed:
   pip install pandas numpy textblob matplotlib
   
2. Run the script:
   python sentiment_analysis.py

3. The script will generate:
- A CSV file with sentiment analysis results
- A PNG image with a visualization of the results

#Output

- `twitter_sentiment.csv`: Contains the generated tweets and their sentiment scores
- `sentiment_analysis.png`: A box plot showing sentiment distribution across topics

This project showcases basic Python programming, data manipulation with pandas, simple NLP techniques, and data visualization skills.




## PROJECT - ENHANCED SENTIMENT ANALYSIS 


This project is an enhanced version of the sentiment analysis tool, incorporating more advanced techniques and visualizations. It demonstrates intermediate to advanced skills in Python, natural language processing (NLP), machine learning, and data visualization.

#Features

- Generates fake Twitter data
- Performs sentiment analysis using Hugging Face Transformers
- Implements basic text preprocessing
- Classifies sentiments into categories (Positive, Negative, Neutral)
- Creates advanced visualizations including word clouds
- Includes a simple unit test

#Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- transformers
- torch
- wordcloud

#Installation

1. Install required libraries:
   pip install pandas numpy matplotlib seaborn transformers torch wordcloud
   
2. Download necessary NLTK data:
  python -m textblob.download_corpora

#Usage

1. Run the script:
python sentiment_analysis_enhanced.py

2. The script will generate:
- A CSV file with detailed sentiment analysis results
- Multiple PNG images with various visualizations

#Output

- `twitter_sentiment.csv`: Contains generated tweets and their sentiment scores
- `sentiment_boxplot.png`: Box plot of sentiment scores by topic
- `sentiment_distribution.png`: Bar plot of sentiment category distribution
- `wordcloud.png`: Word cloud visualization of all tweets

#Testing

The script includes a basic unit test for the sentiment classification function. To run the test:
python -m unittest sentiment_analysis_enhanced.py

This project showcases more advanced Python programming, use of modern NLP libraries, data preprocessing techniques, advanced data visualization, and basic testing practices.
