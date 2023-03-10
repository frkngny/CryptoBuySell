from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


class TModel:
    def __init__(self, train_data, input_data):
        self.time_col = "time"
        self.price_col = "price"
        self.ratio_col = "ls_ratio_overall"
        self.output_data = None
        self.l_ratio = "L_ratio"

        self.train_data = train_data
        self.test_data = input_data

        self.model = None
        self.predicted = None

    def create_train_and_test(self):
        # Read dataset
        data = self.train_data[[self.time_col, self.ratio_col, self.price_col]]
        data = data.sort_values(self.time_col)
        data.index = [i for i in range(len(data.index))]

        data[self.l_ratio] = data[self.ratio_col] / (1 + data[self.ratio_col])

        # load test data
        test_data = self.test_data[[self.time_col, self.ratio_col, self.price_col]]
        test_data = test_data.sort_values(self.time_col)
        test_data.index = [i for i in range(len(data.index) + 1, len(data.index) + 1 + len(test_data.index))]
        test_data[self.l_ratio] = test_data[self.ratio_col] / (1 + test_data[self.ratio_col])

        self.train_data = data
        self.test_data = test_data.iloc[-1:]

    def create_model(self):
        y = self.train_data[self.l_ratio]
        self.model = ARIMA(y, order=(1, 1, 0))  # 807, enforce_stationarity=False, enforce_invertibility=False
        self.model = self.model.fit()

    def create_prediction(self):
        y_pred = self.model.get_forecast(len(self.test_data.index))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df["Predictions"] = self.model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = self.test_data.index
        self.predicted = y_pred_df["Predictions"]

        # arima_rmse = np.sqrt(mean_squared_error(self.test_data[self.l_ratio].values, y_pred_df["Predictions"]))
        # print("RMS: ", arima_rmse)

    def create_buy_sell(self):
        self.create_prediction()
        # self.test_data['Signal'] = np.where(self.test_data[self.l_ratio] > 0.5, 'sell', 'buy')
        self.test_data['Signal'] = np.where(self.test_data[self.l_ratio] > self.predicted, 'sell', 'buy')
        self.test_data.index = [i for i in range(len(self.test_data.index))]

        return self.test_data
