import time
import requests
import json
import pandas as pd


# api key f3b118c38f184e539162436603e2d131


class CoinGlassAPI:
    def __init__(self, ioparams: dict, disabled: list, input_csv: str):
        self.base_url = "https://open-api.coinglass.com/public/v2/"
        self.io_params = ioparams
        self.disabled = disabled
        self.API_KEY = "f3b118c38f184e539162436603e2d131"
        self.final_item = dict()
        self.input_csv = input_csv

    def get_data(self, segment: str, params: dict):
        try:
            symbol = params["symbol"]
        except KeyError:
            symbol = None

        try:
            time_type = params["time_type"]
        except KeyError:
            time_type = None

        if time_type is not None:
            url = f"{self.base_url}/{segment}?time_type={time_type}&symbol={symbol}"
        else:
            url = f"{self.base_url}/{segment}?symbol={symbol}"

        return \
            json.loads(requests.get(url, headers={"accept": "application/json", "coinglassSecret": self.API_KEY}).text)[
                "data"]

    def create_and_write(self, returned_data: list, params: list, segment: str):
        try:
            list_params: list = self.io_params[segment]["list"]
            for key in params:
                if not key.startswith("--"):
                    list_params.append(key)
        except KeyError:
            list_params = list()
            pass

        for data in returned_data:
            if segment == "perpetual_market":
                if data["exchangeName"].lower() != "binance":
                    continue
            if segment == "long_short":
                pass

            item = dict()
            for key in params:
                if key.startswith("--"):
                    key = key[2:]
                try:
                    item[key] = data[key]
                except KeyError as kErr:
                    if kErr.args[0] == "updateTime":
                        item[key] = round(time.time() * 1000)
            if segment == "perpetual_market":
                item = {k: item[k] for k in item.keys() if k == "price" or k == "updateTime"}
                self.final_item["time"] = item["updateTime"]
                self.final_item["price"] = item["price"]
            elif segment == "long_short":
                ls_ratio = item["longRate"] / item["shortRate"]
                self.final_item["ls_ratio_overall"] = ls_ratio

    def prepare_input(self):
        df1 = pd.read_csv(self.input_csv)
        df2 = pd.DataFrame()
        for k in df1.keys():
            try:
                df2[k] = [self.final_item[k]]
            except:
                df2[k] = ""
        try:
            t1 = df1["time"].values[-1]
            t2 = df2["time"].values[-1]
            df2["time_diff_ms"] = [t2 - t1]
        except:
            pass

        df2.to_csv(self.input_csv, mode='a', index=False, header=False)

    def common_market(self):
        for segment, market_class in self.io_params.items():
            if segment not in self.disabled:
                url_params = market_class["url_params"]
                symbol = url_params["symbol"].upper()
                params = market_class["params"]

                returned_data = list()
                flag = False
                if segment == "perpetual_market":
                    returned_data = self.get_data(segment, url_params)[symbol]
                    flag = True
                elif segment == "long_short":
                    returned_data = self.get_data(segment, url_params)
                    flag = True
                # elif segment == "open_interest":
                #     returned_data = self.get_data(segment, url_params)
                # elif segment == "option":
                #     returned_data = self.get_data(segment, url_params)
                # elif segment == "liquidation_history":
                #     returned_data = self.get_data(segment, url_params)
                # else:
                #     return
                if flag:
                    self.create_and_write(returned_data, params, segment)
                    self.final_item["symbol"] = symbol
        self.prepare_input()


if __name__ == "__main__":
    from Utilities import load_io_params

    io_params = load_io_params()
    disabled = [""]
    cg = CoinGlassAPI(io_params, disabled, "input_data2.csv")
    cg.common_market()
