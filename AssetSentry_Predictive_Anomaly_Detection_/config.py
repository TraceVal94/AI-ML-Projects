class Config:
    DATA_PATH = 'data/turbofan_engine_data.csv'
    MODEL_PARAMS = {
        'arima_order': (1, 1, 1),
        'isolation_forest_contamination': 0.1
    }