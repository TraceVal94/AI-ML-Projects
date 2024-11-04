## PORTFOLIO SUMMARY ##
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Welcome to my Artificial Intelligence/Machine Learning project repo! This portfolio includes both personal projects undergone as part of independent proactive learning and 'some' coursework projects undergone as part of my **MSc Computer Science with Artififical Intelligence (Jan 2024 - December 2025. Graduating January 2026)**

## Project 1 - Sentiment Analysis Tool
- #Pandas #Numpy #Textblob #Matplotlib 


## Project 2 - 'Enhanced' Sentiment Analysis Tool
- #Pandas #Numpy #Textblob #Matplotlib #Seaborn #Transformers #Torch #Wordcloud


## Project 3 - SocialGuard - Social Media Compliance Analyzer
- #NTLK #Matplotlib #Scikit-learn #Pandas #Streamlit


## Project 4 - AssetSentry: Predictive Maintenance Tool 
- #pmdarima #statsmodels #Scikit-learn #Pandas #Numpy


## Coursework 1 - Module: Applied AI
- Knapsack Problem - Examination and Implementation of Local Search Algorithms to complex problems
- #SimulatedAnnealing #TabuSearch #Algorithms


## Coursework 2 - Module: Applied AI
- AI/ML NGO Report Casestudy. Examination and Implementation of Machine learning techniques and AI search or optimization techniques to construct a robust model for analysis of datasets
- #GeneticAlgorithms HillClimbingAlgorithm #DecisionTrees #LinearRegression #RMSE #MAE #DataPreprocessing #Prediction #Pandas #Numpy #GridSearch #Matplotlib #Optimisation #ModelEvaluation #ModelTraining 


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# PROJECT 1 - SENTIMENT ANALYSIS TOOL 


This project is a basic sentiment analysis tool that analyzes fake Twitter data. It demonstrates fundamental skills in Python, natural language processing (NLP), and data visualization.

## Features ##

- Generates fake Twitter data
- Performs basic sentiment analysis using TextBlob
- Saves results to a CSV file
- Visualizes results using matplotlib

## Requirements ##

- Python 3.x
- pandas
- numpy
- textblob
- matplotlib

## Usage ##

1. Ensure you have all required libraries installed:
   pip install pandas numpy textblob matplotlib
   
2. Run the script:
   python sentiment_analysis.py

3. The script will generate:
- A CSV file with sentiment analysis results
- A PNG image with a visualization of the results

## Output ##

- `twitter_sentiment.csv`: Contains the generated tweets and their sentiment scores
- `sentiment_analysis.png`: A box plot showing sentiment distribution across topics

This project showcases basic Python programming, data manipulation with pandas, simple NLP techniques, and data visualization skills.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PROJECT 2 - ENHANCED SENTIMENT ANALYSIS 


This project is an enhanced version of the sentiment analysis tool, incorporating more advanced techniques and visualizations. It demonstrates intermediate to advanced skills in Python, natural language processing (NLP), machine learning, and data visualization.

## Features ##

- Generates fake Twitter data
- Performs sentiment analysis using Hugging Face Transformers
- Implements basic text preprocessing
- Classifies sentiments into categories (Positive, Negative, Neutral)
- Creates advanced visualizations including word clouds
- Includes a simple unit test

## Requirements ##

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- transformers
- torch
- wordcloud

## Installation ##

1. Install required libraries:
   pip install pandas numpy matplotlib seaborn transformers torch wordcloud
   
2. Download necessary NLTK data:
  python -m textblob.download_corpora

## Usage ##

1. Run the script:
python sentiment_analysis_enhanced.py

2. The script will generate:
- A CSV file with detailed sentiment analysis results
- Multiple PNG images with various visualizations

## Output ##

- `twitter_sentiment.csv`: Contains generated tweets and their sentiment scores
- `sentiment_boxplot.png`: Box plot of sentiment scores by topic
- `sentiment_distribution.png`: Bar plot of sentiment category distribution
- `wordcloud.png`: Word cloud visualization of all tweets

## Testing ##

The script includes a basic unit test for the sentiment classification function. To run the test:
python -m unittest sentiment_analysis_enhanced.py

This project showcases more advanced Python programming, use of modern NLP libraries, data preprocessing techniques, advanced data visualization, and basic testing practices.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PROJECT 3 - SOCIALGUARD - SOCIAL MEDIA COMPLIANCE ANALYZER


This project demonstrates a tool for analyzing social media posts for policy compliance and sentiment. It uses natural language processing (NLP) and machine learning techniques to classify posts by topic and sentiment, providing insights that could be valuable for companies monitoring their social media presence.

## Features ##

- Generation of fake social media posts for demonstration purposes
- Sentiment analysis of posts (Positive, Negative, Neutral)
- Topic classification of posts (e.g., customer service, product feedback)
- Visualization of sentiment and topic distributions
- Word cloud generation from post content
- Display of sample posts with their classifications
- Evaluation of topic classification performance

## Technologies Used ##

- Python 3.x
- NLTK for sentiment analysis
- Scikit-learn for machine learning (topic classification)
- Pandas for data manipulation
- Matplotlib and WordCloud for data visualization
- Streamlit for the web application interface

## Installation ##

1. Clone this repository:
   git clone https://github.com/TraceVal94/AI-ML-Projects/social-media-policy-analyzer.git
cd social-media-policy-analyzer

2. Install the required packages:
   pip install pandas nltk scikit-learn matplotlib wordcloud streamlit

3. Download the necessary NLTK data:
python
import nltk
nltk.download('vader_lexicon')

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PROJECT 4 - ASSETSENTRY: PREDICTIVE MAINTENANCE TOOL


AssetSentry is a predictive maintenance tool that uses time-series analysis and anomaly detection to predict equipment failures based on sensor data. This project demonstrates skills in data processing, machine learning, and web application development.

## Features ##

* Time-series forecasting using ARIMA models
* Anomaly detection using Isolation Forest algorithm
* Data preprocessing and feature engineering
* Real-time visualization of predictions and anomaly scores
* Web-based dashboard for easy monitoring
* Automated model parameter optimization

## Technologies Used ##

* Python 3.x
* Flask for web application framework
* Pandas and NumPy for data manipulation
* Scikit-learn for machine learning (anomaly detection)
* Statsmodels for time series analysis (ARIMA)
* Matplotlib and Chart.js for data visualization
* pmdarima for automatic ARIMA parameter selection

## Installation ##

1. Clone this repository:
git clone https://github.com/TraceVal94/AI-ML-Projects.git
cd AI-ML-Projects/AssetSentry

2. Install the required packages:
pip install -r requirements.txt

## Usage ##

1. Run the Flask application:
python app.py

2. Open a web browser and navigate to `http://127.0.0.1:5000/`

3. The dashboard will display real-time predictions and anomaly scores for the equipment.

## Data ##

The project uses the NASA Turbofan Engine Degradation Simulation Dataset, which is included in the `data` folder.

## Output ##

* Real-time visualization of time series predictions
* Anomaly score tracking over time
* API endpoint for retrieving latest predictions and anomaly scores

## Model Details ##

* Time Series Forecasting: ARIMA model with automated parameter selection
* Anomaly Detection: Isolation Forest algorithm

## Future Enhancements ##

* Integration with live data streams
* Implementation of more advanced deep learning models (e.g., LSTM)
* Alerting system for detected anomalies
* Extended dashboard features for historical data analysis

This project showcases advanced skills in time series analysis, anomaly detection, data preprocessing, web application development, and real-time data visualization. It demonstrates the ability to create a practical, industry-relevant tool for predictive maintenance.
