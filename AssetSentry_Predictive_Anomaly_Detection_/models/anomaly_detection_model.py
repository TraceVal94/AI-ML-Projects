from sklearn.ensemble import IsolationForest
from config import Config

class AnomalyDetectionModel:
    def __init__(self):
        self.model = IsolationForest(contamination=Config.MODEL_PARAMS['isolation_forest_contamination'])

    def train(self, data):
        self.model.fit(data)

    def detect(self, input_data):
        return self.model.decision_function(input_data)