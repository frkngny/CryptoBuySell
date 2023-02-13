from coinglassApi import CoinGlassAPI
from Utilities import load_io_params
import time
from threading import Thread


class DataHandler:
    def __init__(self, input_csv: str):
        self.io_params = load_io_params()
        self.disabled = [""]  # list to disable to get data for (e.g. "open_interest")

        self.cgApi = CoinGlassAPI(self.io_params, self.disabled, input_csv)
        self.time_interval = 5

    def collect_data_and_store(self):
        while True:
            start_time = time.perf_counter()

            self.cgApi.common_market()  # collect from api and write into db
            end_time = time.perf_counter()
            if (start_time - end_time) > self.time_interval:
                time.sleep(self.time_interval)
            else:
                wait = self.time_interval - (start_time - end_time)
                time.sleep(wait)

    def runner(self):
        col_store = Thread(target=self.collect_data_and_store)
        col_store.start()