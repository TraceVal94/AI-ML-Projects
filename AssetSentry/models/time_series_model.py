from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from config import Config

class TimeSeriesModel:
    def __init__(self):
        self.model = None

    def find_best_parameters(self, data):
        # Perform a grid search to find the best parameters
        model = auto_arima(data, start_p=0, start_q=0, max_p=5, max_q=5, m=1,start_P=0, seasonal=False, d=1, D=1, trace=True,error_action='ignore', suppress_warnings=True, stepwise=True)
        return model.order

    def train(self, data):
        best_order = self.find_best_parameters(data)
        self.model = ARIMA(data, order=best_order)
        self.model_fit = self.model.fit()

    def predict(self, steps=1):
        return self.model_fit.forecast(steps=steps)