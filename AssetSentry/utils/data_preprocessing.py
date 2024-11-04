import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

def make_stationary(data):
    result = adfuller(data)
    if result[1] > 0.05:  # p-value > 0.05 indicates non-stationary
        diff_data = np.diff(data)
        return diff_data
    return data

def preprocess_data(data):
    # Define column names
    columns = ['unit', 'cycle'] + [f'sensor{i}' for i in range(1, 22)] + ['setting1', 'setting2', 'setting3']
    
    # Read data and assign column names
    df = pd.read_csv(data, sep='\s+', header=None, names=columns)
    
    # Select relevant features
    features = ['cycle', 'sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor11', 'sensor12', 'sensor15', 'setting1', 'setting2', 'setting3']
    df_selected = df[features]
    
    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected)
    
    return pd.DataFrame(scaled_data, columns=features)

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)