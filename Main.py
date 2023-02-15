import time

from MModel import InitModel
from DataHandler import DataHandler
from threading import Thread


class Main:
    def __init__(self):
        self.time_col = "time"
        self.price_col = "price"
        self.ratio_col = "ls_ratio_overall"
        self.l_ratio = "L_ratio"
        self.train_csv = "training_data2.csv"
        self.test_csv = "input_data2.csv"
        self.output_csv = "output2.csv"
        self.initModel = InitModel(self.time_col, self.price_col, self.ratio_col, self.train_csv, self.test_csv,
                                   self.output_csv)
        self.dataMngr = DataHandler(self.test_csv)
        self.time_interval = 5

    def common_process(self):
        self.initModel.create_train_and_test()
        self.initModel.create_model()
        self.initModel.create_profit_factor()

    def init_process(self):
        ctr = True
        while True:
            try:
                start_time = time.perf_counter()

                self.dataMngr.collect_data_and_store()
                if ctr:
                    self.async_proc()
                else:
                    Thread(target=self.async_proc_model()).start()
                    self.async_proc_signal()

                end_time = time.perf_counter()
                if (start_time - end_time) > self.time_interval:
                    time.sleep(self.time_interval)
                else:
                    wait = self.time_interval - (start_time - end_time)
                    time.sleep(wait)
            except:
                continue
            ctr = False

    def async_proc_model(self):
        self.initModel.create_train_and_test()
        self.initModel.create_model()

    def async_proc_signal(self):
        self.initModel.create_buy_sell()
        self.initModel.create_profit_factor()

    def async_proc(self):
        """
        Continuously run this.
        """
        self.initModel.create_train_and_test()
        self.initModel.create_model()
        self.initModel.create_buy_sell()
        self.initModel.create_profit_factor()

    def start_process(self):
        thrd = Thread(target=self.init_process())
        thrd.start()


if __name__ == "__main__":
    main = Main()
    main.init_process()
