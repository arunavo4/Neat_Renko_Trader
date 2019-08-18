from os import path
import gym
import time
import multiprocessing as mp
from statistics import mean
from lib.env.TraderEnv import StockTradingEnv
from lib.features.indicators import add_indicators
from utils.cache import get_observations
import pandas as pd
from utils.data_utils import *

# # logging
# # importing module
# import logging
#
# formatter = logging.Formatter('%(message)s')
#
#
# def setup_logger(name, log_file, level=logging.INFO):
#     """Function setup as many loggers as you want"""
#
#     handler = logging.FileHandler(log_file)
#     handler.setFormatter(formatter)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#
#     return logger
#
#
# Test_logger = setup_logger('env_logger', 'env_test.log')
# Test_logger.info('Env_Test Logger!')


# The data with obs and normalized

filename = '../data/dataset/small/ADANIPORTS-EQ.csv'

df = load_csv(filename)
df = add_indicators(df.reset_index())

train_df, test_df = split_data(df)

train_df = train_df[:5000]

params = {
        'look_back_window_size': 120,
        'enable_stationarization': True,
        'n_processes': mp.cpu_count(),
        'pre_computed_observation': False
    }

start_offset = int(375)

if int(params['look_back_window_size'] / 375) > 1:
    start_offset *= int(params['look_back_window_size'] / 375)

max_env_steps = len(train_df) - start_offset - 1

# obs_dict = get_observations(train_df, params, False)
obs_dict = None

env = StockTradingEnv(train_df,obs_dict, **params)

observation = env.reset()

time_obs = []

for i in range(375):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)

    start = time.time()
    observation, reward, done, info = env.step(action)
    end = time.time()

    time_obs.append(end-start)
    # print("###############################")
    # print("Step:", env.current_step)
    # print(action)
    # print(str(observation) + str(reward) + str(done) + str(info))
    # print(len(observation))
    # print("###############################")

    if done:
        observation = env.reset()
env.close()

print("Avg Response Time: ", mean(time_obs))
print("Theoretical Traversal Time: {} Min".format((mean(time_obs)*max_env_steps)/60))
