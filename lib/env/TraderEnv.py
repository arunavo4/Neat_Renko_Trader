"""
    # This Env version has Pre-Computed Observations
    # compute heavy nature.
"""

from collections import deque
from statistics import mean
import math
import gym
import time
import pandas as pd
import numpy as np
from gym import spaces
from lib.features.indicators import get_pattern_columns
from lib.render.StockTradingGraph import StockTradingGraph
from lib.features.transform import log_and_difference, max_min_normalize

# logging
import logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Func to calculate brokerage
def cal_profit_w_brokerage(buy_price, sell_price, qty):
    turnover = (buy_price * qty) + (sell_price * qty)
    brokerage = ((0.01 / 100) * turnover)
    stt = math.ceil((0.025 / 100) * (sell_price * qty))
    exn_trn = ((0.00325 / 100) * turnover)
    gst = (18 / 100) * (brokerage + exn_trn)
    sebi = (0.000001 * turnover)

    return ((sell_price - buy_price) * qty) - (brokerage + stt + exn_trn + gst + sebi)


# Delete this if debugging
np.warnings.filterwarnings('ignore')


class StockTradingEnv(gym.Env):
    '''A Stock trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, observation_dict=None, initial_balance=10000, **kwargs):
        super(StockTradingEnv, self).__init__()

        self.initial_balance = initial_balance

        self.enable_logging = kwargs.get('enable_env_logging', True)
        if self.enable_logging:
            self.logger = setup_logger('env_logger', 'env.log')
            self.logger.info('Env Logger!')

        # Stuff from the Env Before
        self.decay_rate = 1e-2
        self.done = False
        self.current_step = int(375)
        self.t = int(0)
        self.wins = int(0)
        self.losses = int(0)
        self.qty = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.amt = self.initial_balance
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        self.df = df.fillna(method='bfill').reset_index()
        self.stationary_df = self.df.copy()

        self.pre_computed_observation = kwargs.get('pre_computed_observation', False)
        if self.pre_computed_observation:
            if observation_dict is None:
                raise ValueError('Observation Dict Missing!')
            else:
                self.observation_dict = observation_dict

        self.enable_stationarization = kwargs.get('enable_stationarization', True)
        self.look_back_window_size = kwargs.get('look_back_window_size', 120)
        n_features = 5 + len(self.df.columns) - 1
        self.obs_shape = (1, n_features)

        # Actions of the format Buy, Sell , Hold .
        self.action_space = spaces.Discrete(3)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def resetReward(self):
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0

    def getReward(self, reward):
        stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)
        return self.sum / self.denominator

    def set_qty(self, price):
        self.qty = int(self.balance / price)

        if self.qty == 0:
            self.done = True

    def _next_observation(self):
        current_idx = self.current_step + 1

        if self.pre_computed_observation:
            obs = self.observation_dict[current_idx]

        else:
            observations = self.stationary_df[-self.look_back_window_size + current_idx:current_idx].values

            observations = pd.DataFrame(observations, columns=self.stationary_df.columns)

            observations = observations[observations.columns.difference(['index', 'Date'])]

            if self.enable_stationarization:
                scaled = log_and_difference(observations, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            else:
                scaled = observations

            scaled = max_min_normalize(scaled, columns=scaled.columns, pattern_columns=get_pattern_columns(),
                                       pattern_normalize=True)
            obs = scaled.values[-1]

        scaled_history = max_min_normalize(self.account_history[-self.look_back_window_size:])

        obs = np.insert(obs, len(obs), scaled_history.values[-1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs[0]

    def _current_price(self):
        return self.df['Close'].values[self.current_step]

    def _current_timestamp(self):
        return self.df['Date'].values[self.current_step]

    def _current_index(self):
        return self.df.index[self.current_step]

    def _take_action(self, action):
        current_price = self._current_price()

        action_type = int(action)  # [0, 1, 2]

        reward = 0
        profit_percent = float(0)
        self.position_record = ""
        self.action_record = ""
        # set next time
        self.t = self._current_index()
        if (self.t + 10) % 375 == 0:
            # auto square-off at 3:20 pm and skip to next day
            # Check of trades taken
            self.tradable = False
            if len(self.positions) != 0:
                if self.enable_logging:
                    self.logger.info("{} Auto Square-Off".format(self._current_timestamp()))
                if self.short:
                    action_type = 1
                else:
                    action_type = 2

        # act = 0: hold, 1: buy, 2: sell
        if action_type == 1 and self.market_open:  # buy
            if len(self.positions) == 0:
                if self.tradable:
                    # Going Long
                    self.positions.append(float(current_price))
                    self.set_qty(float(current_price))
                    reward = 1
                    self.short = False
                    self.balance -= self.positions[-1] * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Long",
                                                                   round(reward, 3))
                    self.trades.append({'step': self.current_step,
                                        'amount': self.qty,
                                        'total': (self.qty * float(current_price)),
                                        'type': 'buy'})
                    self.action_record = message
                    if self.enable_logging:
                        self.logger.info(message)

            elif not self.short and len(self.positions) != 0:
                # If stock has been already long
                reward = 0
                message = "{}: Don't try to go long more than once!".format(self._current_timestamp())
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                # exit from Short Sell
                profits = 0
                profits += cal_profit_w_brokerage(float(current_price), mean(self.positions),
                                                  self.qty)

                avg_profit = profits / self.qty
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt * (profit_percent / 100)
                self.balance = self.amt
                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                if profit_percent > 0.3 or profit_percent < -0.3:
                    reward += self.getReward(profit_percent) * 300
                else:
                    reward += self.getReward(profit_percent) * 30
                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty * -1,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                self.resetReward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Short",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.trades.append({'step': self.current_step,
                                    'amount': self.qty,
                                    'total': (self.qty * float(current_price)),
                                    'type': 'buy'})
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        elif action_type == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
                    if self.short:
                        profits += (p - float(current_price))
                    else:
                        profits += (float(current_price) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
                message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Hold",
                                                               round(reward, 3))
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                self.action_record = "Thinking for next move!" if self.market_open else "##-------------##"
                message = "{}: {}".format(self._current_timestamp(), self.action_record)
                if self.enable_logging:
                    self.logger.info(message)

        elif action_type == 2 and self.market_open:  # sell
            if len(self.positions) == 0:
                # Going Short
                if self.tradable:
                    self.positions.append(float(current_price))
                    self.set_qty(float(current_price))
                    reward = 1
                    self.short = True
                    self.balance -= self.positions[-1] * self.qty
                    message = "{}: Action: {} ; Reward: {}".format(self._current_timestamp(), "Short",
                                                                   round(reward, 3))
                    self.trades.append({'step': self.current_step,
                                        'amount': self.qty,
                                        'total': (self.qty * float(current_price)),
                                        'type': 'sell'})
                    self.action_record = message
                    if self.enable_logging:
                        self.logger.info(message)

            elif self.short and len(self.positions) != 0:
                # If stock has been already short
                reward = 0
                message = "{}: Don't try to short more than once!".format(self._current_timestamp())
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(message)

            else:
                # exit from the Long position
                profits = 0
                profits += cal_profit_w_brokerage(mean(self.positions), float(current_price),
                                                  self.qty)

                avg_profit = profits / self.qty
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt * (profit_percent / 100)
                self.balance = self.amt
                if profit_percent > 0.0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.profit_per += round(profit_percent, 3)
                if profit_percent > 0.3 or profit_percent < -0.3:
                    reward += self.getReward(profit_percent) * 300
                else:
                    reward += self.getReward(profit_percent) * 30
                # Save the record of exit
                self.position_record = "{}: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self._current_timestamp(),
                    self.qty,
                    round(mean(self.positions), 2),
                    round(float(current_price), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                self.resetReward()
                self.qty = int(0)
                message = "{}: Action: {} ; Reward: {} ; Profit_Per: {}".format(self._current_timestamp(),
                                                                                "Exit Long",
                                                                                round(reward, 3),
                                                                                round(profit_percent, 2))
                self.trades.append({'step': self.current_step,
                                    'amount': self.qty,
                                    'total': (self.qty * float(current_price)),
                                    'type': 'sell'})
                self.action_record = message
                if self.enable_logging:
                    self.logger.info(self.position_record)
                    self.logger.info(message)

        if (self.t + 10) % 375 == 0:
            # Close Market at 3:20 pm and skip to next day
            if self.enable_logging:
                self.logger.info("{} Market Closed".format(self._current_timestamp()))
            self.market_open = False

        if (self.t + 1) % 375 == 0:
            self.market_open = True
            self.tradable = True
            reward += self.profit_per * 1000  # Bonus for making a profit at the end of the day
            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0
            # Reset Profits for the day
            self.profits = 0.0

        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(current_price)) * self.qty
        self.net_worths.append(self.balance + self.position_value)

        self.account_history = self.account_history.append({
            'timestep': ((self.t + 1) % 375),
            'net_worth': round(self.net_worths[-1], 2),
            'stock_qty': self.qty,
            'position_value': round(self.position_value, 2),
            'profits': round(self.profits, 2),
            'profits_per': round(self.profit_per, 3),
        }, ignore_index=True)
        if self.market_open:
            if self.enable_logging:
                self.logger.info(
                    "{}: {} Net_worth: {} Stk_Qty: {} Pos_Val: {} Profits: {} Profit_Per: {}".format(
                        self._current_timestamp(),
                        ((self.t + 1) % 375),
                        round(self.net_worths[-1], 2),
                        round(self.qty),
                        round(self.position_value, 2),
                        round(self.profits, 2),
                        round(self.profit_per, 3), ))

        # clip reward
        reward = round(reward, 3)

        return reward

    def _reward(self):
        length = self.current_step
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0
        reward = 0

        return reward if np.isfinite(reward) else 0

    def _done(self):
        self.done = self.net_worths[-1] < self.initial_balance / 5 or self.current_step == len(self.df) - 1

    def reset(self):
        self.balance = self.initial_balance
        self.stock_held = 0

        if int(self.look_back_window_size / 375) > 1:
            self.current_step = int(375) * int(self.look_back_window_size / 375)
        else:
            self.current_step = int(375)

        self.net_worths = [self.initial_balance] * (self.current_step + 1)
        self.initial_step = self.current_step
        self.done = False
        self.t = int(0)
        self.wins = int(0)
        self.losses = int(0)
        self.short = False
        self.tradable = True
        self.market_open = True
        self.amt = self.initial_balance
        self.qty = int(0)
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

        self.account_history = pd.DataFrame([{
            'timestep': 0,
            'net_worth': self.balance,
            'stock_qty': 0,
            'position_value': 0,
            'profits': 0,
            'profits_per': 0,
        }])
        self.trades = []

        return self._next_observation()

    def step(self, action):
        reward = self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        # reward = self._reward()
        self._done()

        return obs, reward, self.done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('\t\t' + '=' * 20 + ' History ' + '=' * 20)
            print(str(self.account_history[-1:]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = StockTradingGraph(self.df)

            self.viewer.render(self.current_step, self.net_worths, self.initial_step, self.trades, window_size=20)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
