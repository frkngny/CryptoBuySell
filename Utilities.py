import json


def buy_at(price):
    return


def sell_at(price):
    return


def calculate_profit_factor(buys, price):
    profits = [price - buy_price for buy_price in buys]
    plus = [p if p > 0 else 0 for p in profits]
    minus = [p if p < 0 else 0 for p in profits]
    if sum(minus) == 0:
        profit_factor = sum(plus)
    else:
        profit_factor = sum(plus) / sum(minus)

    return profit_factor, profits


def load_io_params():
    with open('io_params.json') as json_file:
        return json.load(json_file)
