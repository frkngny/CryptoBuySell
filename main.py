import datetime
import time

import pandas as pd
from TModel import TModel
from DataHandler import DataHandler
from Trader import Trader
from Utilities import check_create_files


class Main:
    def __init__(self):
        self.time_col = "time"
        self.price_col = "price"
        self.ratio_col = "ls_ratio_overall"
        self.l_ratio = "L_ratio"

        self.train_csv = "training_data_test.csv"
        self.input_csv = "input_data_test.csv"
        self.output_csv = "output_stored_test.csv"

        self.trader = Trader(self.output_csv)
        self.dataMngr = DataHandler(self.input_csv)
        self.time_interval = 5

        self.model = None
        self.signaled_data = None

        self.training_data = None
        self.training_decider_data = None
        self.input_data = None

    def train_and_pred(self):
        self.model = TModel(self.training_data, self.input_data)
        self.model.create_train_and_test()
        self.model.create_model()
        self.update_train_data()
        return self.model.create_buy_sell()

    def update_train_data(self):
        self.training_data = pd.concat([self.training_data, self.input_data])
        self.training_data.to_csv(self.train_csv, index=False)

    def create_data(self):
        # training data
        data = pd.read_csv(self.train_csv)
        if len(data) > 5000:
            data = data[-5000:]

        data = data[[self.time_col, self.ratio_col, self.price_col]]
        data = data.sort_values(self.time_col)
        data.index = [i for i in range(len(data.index))]

        data[self.l_ratio] = data[self.ratio_col] / (1 + data[self.ratio_col])
        self.training_data = data
        self.training_decider_data = data[-50:]

        # input data
        input_data = pd.read_csv(self.input_csv)
        input_data = input_data[[self.time_col, self.ratio_col, self.price_col]]
        input_data = input_data.sort_values(self.time_col)
        input_data.index = [i for i in range(len(data.index) + 1, len(data.index) + 1 + len(input_data.index))]
        input_data[self.l_ratio] = input_data[self.ratio_col] / (1 + input_data[self.ratio_col])
        self.input_data = input_data.iloc[-1:]

    def trade(self, input_data):
        self.trader.start_trade(input_data, self.training_decider_data)

    def init_process(self):
        # check_create_files(self.train_csv, self.input_csv, self.output_csv)
        #
        # dataMngr2 = DataHandler(self.train_csv)
        # for pt in range(5000):
        #
        #     try:
        #         dataMngr2.collect_data_and_store()
        #     except:
        #         continue
        #
        #     time.sleep(1)

        while True:
            try:
                self.dataMngr.collect_data_and_store()

                self.create_data()

                self.signaled_data = self.train_and_pred()
                stored = pd.read_csv(self.output_csv)
                stored = pd.concat([stored, self.signaled_data])
                stored.to_csv(self.output_csv, index=False)

                self.trade(self.signaled_data)
            except:
                print("Long: ", self.trader.long_stocks)
                print("Short: ", self.trader.short_stocks)
                print("Budget: ", self.trader.budget)
                print("Profit: ", self.trader.total_profit)
                print("P Profits: ", self.trader.plus_profits)
                print("M Profits: ", self.trader.minus_profits)
                try:
                    print("Profit Factor: ", abs(self.trader.plus_profits / self.trader.minus_profits))
                except:
                    pass

    # -------------- TESTING

    def import_test_data(self, j):
        df1 = pd.read_csv("import_from.csv").loc[j]
        df2 = pd.read_csv(self.input_csv)
        df3 = pd.DataFrame()
        for k in df2.keys():
            df3[k] = [df1[k]]

        df3.to_csv(self.input_csv, mode='a', index=False, header=False)

    def test_process(self):
        df1 = pd.read_csv("import_from.csv")
        ind = 0
        while ind < len(df1["time"]):
            self.import_test_data(ind)

            self.create_data()

            self.signaled_data = self.train_and_pred()
            stored = pd.read_csv(self.output_csv)
            stored = pd.concat([stored, self.signaled_data])
            stored.to_csv(self.output_csv, index=False)

            self.trade(self.signaled_data)

            ind += 1

        print("Long: ", self.trader.long_stocks)
        print("Short: ", self.trader.short_stocks)
        print("Budget: ", self.trader.budget)
        print("Profit: ", self.trader.total_profit)
        print("P Profits: ", self.trader.plus_profits)
        print("M Profits: ", self.trader.minus_profits)
        print("Profit Factor: ", abs(self.trader.plus_profits/self.trader.minus_profits))


start = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

mn = Main()
mn.init_process()
print("Start: ", start)
print("End: ", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
