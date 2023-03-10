from Utilities import *
import numpy as np
import copy
import pandas as pd


class Signals:
    BUY = "buy"
    SELL = "sell"


class Trader:
    def __init__(self, stored_csv):
        self.time_col = "time"
        self.price_col = "price"
        self.ratio_col = "ls_ratio_overall"
        self.position_col = "position"
        self.l_ratio = "L_ratio"
        self.signal_col = "Signal"
        self.stored_data_csv = stored_csv

        self.test_data = None
        self.output_data = None
        self.model_train = None
        self.predicted = None

        self.long_stocks = {"times": [], "prices": []}
        self.short_stocks = {"times": [], "prices": []}

        self.budget = 500  # edit this initial budget to buy
        self.stop_loss = 0.995
        self.pf = 1.5
        self.total_profit = 0
        self.buy_percent = 0.001

        self.plus_profits = 0
        self.minus_profits = 0

    def start_trade(self, current_data, decider_data):
        ma_50 = self.calc_ma(decider_data, 50).iloc[-1]
        ma_21 = self.calc_ma(decider_data[-21:], 21).iloc[-1]
        ma_9 = self.calc_ma(decider_data, 9).iloc[-1]

        temp_decider = copy.deepcopy(decider_data)
        temp_decider.drop(temp_decider.tail(1).index, inplace=True)
        ma_8 = self.calc_ma(temp_decider, 9).iloc[-1]

        data = copy.deepcopy(current_data)

        data_time = data[self.time_col][0]
        data_price = data[self.price_col][0] * self.buy_percent
        data_signal = data[self.signal_col][0]
        data_l_ratio = data[self.ratio_col][0] / (1 + data[self.ratio_col][0])

        print(f"Current time: {data_time}, price: {data_price}, signal: {data_signal}")
        flag = False
        if data_signal == Signals.BUY:
            cond = ma_21 <= data_l_ratio
            if data_price <= self.budget and cond:
                self.buy_long(data_price, data_time)
                flag = True

            if len(self.short_stocks["prices"]) > 0:
                profit_factor, profits = calculate_profit_factor(self.short_stocks["prices"], data_price, True)
                if profit_factor >= self.pf:
                    short_copy = copy.deepcopy(self.short_stocks)
                    profits_copy = profits.copy()
                    profit_mean = np.average(profits)
                    for k in range(len(profits)):
                        if profits[k] >= profit_mean or (profits[k] < 0 and abs(profits[k]) < profit_mean):
                            flag = True
                            short_copy, profits_copy = self.sell_short(data_price, data_time, short_copy,
                                                                       profits, profits_copy, k)
                    self.short_stocks = short_copy
        else:
            if len(self.long_stocks["prices"]) > 0:
                profit_factor, profits = calculate_profit_factor(self.long_stocks["prices"], data_price)
                if profit_factor >= self.pf:
                    long_copy = copy.deepcopy(self.long_stocks)
                    profits_copy = profits.copy()
                    profit_mean = np.average(profits)
                    for k in range(len(profits)):
                        if profits[k] >= profit_mean or (profits[k] < 0 and abs(profits[k]) < profit_mean):
                            flag = True
                            long_copy, profits_copy = self.sell_long(data_price, data_time, long_copy, profits,
                                                                     profits_copy, k)
                    self.long_stocks = long_copy
            cond1 = ma_21 > data_l_ratio
            if cond1 and data_price <= self.budget:
                flag = True
                self.buy_short(data_price, data_time)

        for q in range(len(self.long_stocks["prices"])):
            if data_price / self.long_stocks["prices"][q] <= self.stop_loss:
                self.long_cut_loss(data_price, q, data_time)

        for q in range(len(self.short_stocks["prices"])):
            if self.short_stocks["prices"][q] / data_price <= self.stop_loss:
                self.short_cut_loss(data_price, q, data_time)

        if flag:
            print(f"Longs: {self.long_stocks}")
            print(f"Shorts: {self.short_stocks}")
            print(f"Budget: {self.budget}")
            print(f"Profit: {self.total_profit}")
            print(f"P Profit: {self.plus_profits}")
            print(f"M Profit: {self.minus_profits}")

    def calc_ma(self, decider: pd.DataFrame, period: int):
        return decider[self.l_ratio].rolling(window=period).mean()

    def buy_long(self, p, t):
        print(f"{BGColors.BLUE}Buy long {p} at {t} {BGColors.ENDC}")
        self.long_stocks["prices"].append(p)
        self.long_stocks["times"].append(t)
        self.budget -= p

    def sell_long(self, p, t, bc, prf, pc, i):
        print(f"{BGColors.HBLUE}Sell long {self.long_stocks['prices'][i]} at {p} with profit: {prf[i]} {BGColors.ENDC}")

        stored_data = pd.read_csv(self.stored_data_csv)
        stored_data.loc[stored_data[self.time_col] == self.long_stocks['times'][i], "Sold_At"] = int(t)
        stored_data.loc[stored_data[self.time_col] == self.long_stocks['times'][i], self.position_col] = "long"
        stored_data.to_csv(self.stored_data_csv, index=False)

        bc['prices'].remove(self.long_stocks['prices'][i])
        bc['times'].remove(self.long_stocks['times'][i])

        pc.remove(prf[i])
        self.total_profit += prf[i]
        self.budget += p
        if prf[i] > 0:
            self.plus_profits += prf[i]
        else:
            self.minus_profits += prf[i]
        return bc, pc

    def long_cut_loss(self, p, i, t):
        print(f"{BGColors.YELLOW}Long Cut loss == bought: {self.long_stocks['prices'][i]}, sold: {p}, "
              f"loss: {self.long_stocks['prices'][i] - p} {BGColors.ENDC}")

        stored_data = pd.read_csv(self.stored_data_csv)
        stored_data.loc[stored_data[self.time_col] == self.long_stocks['times'][i], "Sold_At"] = int(t)
        stored_data.loc[stored_data[self.time_col] == self.long_stocks['times'][i], self.position_col] = "long"
        stored_data.to_csv(self.stored_data_csv, index=False)

        self.total_profit -= (self.long_stocks['prices'][i] - p)
        self.budget += p
        self.minus_profits -= (self.long_stocks['prices'][i] - p)

        self.long_stocks['prices'].pop(i)
        self.long_stocks['times'].pop(i)

        if (self.long_stocks['prices'][i] - p) > 0:
            self.plus_profits += (self.long_stocks['prices'][i] - p)
        else:
            self.minus_profits += (self.long_stocks['prices'][i] - p)

    def buy_short(self, p, t):
        print(f"{BGColors.GREEN}Buy short {p} at {t} {BGColors.ENDC}")

        self.short_stocks['prices'].append(p)
        self.short_stocks['times'].append(t)
        self.budget -= p

    def sell_short(self, p, t, bc, prf, pc, i):
        print(f"{BGColors.HGREEN}Sell short {self.short_stocks['prices'][i]} at {p} "
              f"with profit: {prf[i]} {BGColors.ENDC}")

        stored_data = pd.read_csv(self.stored_data_csv)
        stored_data.loc[stored_data[self.time_col] == self.short_stocks['times'][i], "Sold_At"] = int(t)
        stored_data.loc[stored_data[self.time_col] == self.short_stocks['times'][i], self.position_col] = "short"
        stored_data.to_csv(self.stored_data_csv, index=False)

        bc['prices'].remove(self.short_stocks['prices'][i])
        bc['times'].remove(self.short_stocks['times'][i])

        pc.remove(prf[i])
        self.total_profit += prf[i]
        self.budget += prf[i] + self.short_stocks['prices'][i]

        if prf[i] > 0:
            self.plus_profits += prf[i]
        else:
            self.minus_profits += prf[i]

        return bc, pc

    def short_cut_loss(self, p, i, t):
        print(f"{BGColors.RED}Short Cut loss == bought: {self.short_stocks['prices'][i]}, sold: {p}, "
              f"loss: {self.short_stocks['prices'][i] - p} {BGColors.ENDC}")

        stored_data = pd.read_csv(self.stored_data_csv)
        stored_data.loc[stored_data[self.time_col] == self.short_stocks['times'][i], "Sold_At"] = int(t)
        stored_data.loc[stored_data[self.time_col] == self.short_stocks['times'][i], self.position_col] = "short"
        stored_data.to_csv(self.stored_data_csv, index=False)

        self.total_profit += (self.short_stocks['prices'][i] - p)
        self.budget += p
        self.minus_profits += (self.short_stocks['prices'][i] - p)

        self.short_stocks['prices'].pop(i)
        self.short_stocks['times'].pop(i)
