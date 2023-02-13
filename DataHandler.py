from coinglassApi import CoinGlassAPI
from Utilities import load_io_params


class DataHandler:
    def __init__(self, input_csv: str):
        self.io_params = load_io_params()
        self.disabled = [""]  # list to disable to get data for (e.g. "open_interest")

        self.cgApi = CoinGlassAPI(self.io_params, self.disabled, input_csv)

    def collect_data_and_store(self):
        self.cgApi.common_market()  # collect from api and write into db
