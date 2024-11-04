from flask import Flask, render_template, jsonify, request
from models.time_series_model import TimeSeriesModel
from models.anomaly_detection_model import AnomalyDetectionModel
from utils.data_preprocessing import preprocess_data, create_sequences, make_stationary
from config import Config
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and preprocess data
data = preprocess_data(Config.DATA_PATH)
sequence_length = 50
sequences = create_sequences(data.values, sequence_length)
stationary_data = make_stationary(data['cycle'])

# Initialize and train models
time_series_model = TimeSeriesModel()
time_series_model.train(stationary_data)

anomaly_detection_model = AnomalyDetectionModel()
anomaly_detection_model.train(sequences.reshape(sequences.shape[0], -1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Use the last sequence in our data
    last_sequence = sequences[-1]
    
    # Time series prediction
    time_series_prediction = time_series_model.predict(steps=1)[0]
    
    # Anomaly detection
    anomaly_score = anomaly_detection_model.detect(last_sequence.reshape(1, -1))[0]
    
    return jsonify({
        'prediction': float(time_series_prediction),
        'anomaly_score': float(anomaly_score)
    })

if __name__ == '__main__':
    app.run(debug=True)