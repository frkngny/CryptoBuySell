import copy

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from Utilities import *


class InitModel:
    def __init__(self, time_col: str = "time", price_col: str = "price", ratio_col: str = "ls_overall_ratio",
                 train_csv: str = "data.csv", test_csv: str = "input.csv",
                 output_csv: str = "output.csv"):
        self.time_col = time_col
        self.price_col = price_col
        self.ratio_col = ratio_col
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.output_csv = output_csv
        self.l_ratio = "L_ratio"
        self.train_data = None
        self.test_data = None
        self.output_data = None
        self.model_train = None
        self.model = None
        self.predicted = None
        self.buys = {"times": [], "prices": []}
        self.buy_percents = []
        self.sells = []
        self.budget = 250000  # edit this initial budget to buy
        self.pf = 1.5
        self.total_profit = 0
        self.output_stored = "output_stored.csv"

    def create_train_and_test(self):
        # Read dataset
        data = pd.read_csv(self.train_csv)
        data = data[[self.time_col, self.ratio_col, self.price_col]]
        data = data.sort_values(self.time_col)
        data.index = [i for i in range(len(data.index))]

        data[self.l_ratio] = data[self.ratio_col] / (1 + data[self.ratio_col])

        # load test data
        test_data = pd.read_csv(self.test_csv)
        test_data = test_data[[self.time_col, self.ratio_col, self.price_col]]
        test_data = test_data.sort_values(self.time_col)
        test_data.index = [i for i in range(len(data.index) + 1, len(data.index) + 1 + len(test_data.index))]
        test_data[self.l_ratio] = test_data[self.ratio_col] / (1 + test_data[self.ratio_col])

        self.train_data = data
        self.test_data = test_data.iloc[-1:]

    def create_model(self):
        y = self.train_data[self.l_ratio]
        self.model_train = ARIMA(y, order=(8, 0, 7))
        self.model_train = self.model_train.fit()
        self.model = copy.deepcopy(self.model_train)
        self.update_train_data()

    def create_prediction(self):
        y_pred = self.model.get_forecast(len(self.test_data.index))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df["Predictions"] = self.model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = self.test_data.index
        self.predicted = y_pred_df["Predictions"]
        arma_rmse = np.sqrt(mean_squared_error(self.test_data[self.l_ratio].values, y_pred_df["Predictions"]))
        print(arma_rmse)

    def create_buy_sell(self):
        self.create_prediction()
        self.test_data['Signal'] = np.where(self.test_data[self.l_ratio] > self.predicted, 'sell', 'buy')
        self.test_data.index = [i for i in range(len(self.test_data.index))]
        self.test_data.to_csv(self.output_csv, index=False)

        stored = pd.read_csv(self.output_stored)
        stored = pd.concat([stored, self.test_data])
        stored.to_csv(self.output_stored, index=False)

    def update_train_data(self):
        self.train_data = pd.concat([self.train_data, self.test_data])
        self.train_data.to_csv(self.train_csv, index=False)

    def create_profit_factor(self):
        data = pd.read_csv(self.output_csv)
        stored_data = pd.read_csv(self.output_stored)

        for i in range(len(data)):
            data_price = data[self.price_col][i]
            data_time = data[self.time_col][i]
            data_signal = data['Signal'][i]
            if data_signal == "buy":
                if data_price < self.budget:
                    self.buys["prices"].append(data_price)
                    self.buys["times"].append(data_time)
                    self.budget -= data_price
            else:
                if len(self.buys["prices"]) > 0:
                    profit_factor, profits = calculate_profit_factor(self.buys["prices"], data_price)
                    if profit_factor >= self.pf:
                        buy_copy = copy.deepcopy(self.buys)
                        profits_copy = profits.copy()
                        for k in range(len(profits)):
                            profit_mean = np.average(profits)
                            if profits[k] >= profit_mean:
                                print(f"Sell {self.buys['prices'][k]} at {data_price} with profit: {profits[k]}")
                                stored_data.loc[
                                    stored_data[self.time_col] == self.buys['times'][k], "Sold_At"] = int(data_time)
                                buy_copy['prices'].remove(self.buys['prices'][k])
                                buy_copy['times'].remove(self.buys['times'][k])

                                profits_copy.remove(profits[k])
                                self.total_profit += profits[k]
                                self.budget += data_price
                        self.buys = buy_copy
                        self.sells.append(data_price)

                        print("Total profit: ", self.total_profit)
                        print("Current buy prices: ", self.buys)
                        print("Current budget: ", self.budget)

        stored_data.to_csv(self.output_stored, index=False)
