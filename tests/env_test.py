import multiprocessing
from os import path
from PIL import Image
import cv2
import gym
import time
from statistics import mean
from lib.env.TraderRenkoEnv import StockTradingEnv
from lib.features.indicators import add_indicators
from utils.cache import get_observations
import pandas as pd
import numpy as np
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

filename = '../data/dataset/ADANIPORTS-EQ.csv'

df = load_csv(filename)
# df = add_indicators(df.reset_index())

train_df, test_df = split_data(df)

params = {
    'look_back_window_size': 375*15,
    'enable_stationarization': True,
    'n_processes': multiprocessing.cpu_count(),
    'pre_computed_observation': False,
    'enable_env_logging': False,
    'observation_window': 100
}

max_env_steps = len(train_df) - 1

# obs_dict = get_observations(train_df, params, False)
obs_dict = None

env = StockTradingEnv(train_df,obs_dict, **params)

observation = env.reset()

time_obs = []

frames = []

for i in range(max_env_steps):
    # env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)

    # frames.append(Image.fromarray(observation))
    # path = '../genome_plots/'
    #
    # img = Image.fromarray(observation)
    # img.save(path + str(env.current_step) + '.png')
    #
    # obs = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    #
    # img_grey = Image.fromarray(obs)
    # img_grey.save(path + str(env.current_step) + '_gray.png')
    #
    # array = np.ndarray.flatten(obs)
    print(len(observation))

    # env.plot_renko(path=path)

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
        break
env.close()

# path = '../genome_plots/'
# frames[0].save('plot.gif',
#                save_all=True,
#                append_images=frames[1:],
#                duration=1,
#                loop=0)

print("Avg Response Time: ", mean(time_obs))
# print("Theoretical Traversal Time: {} Min".format((mean(time_obs)*max_env_steps)/60))
print(len(time_obs))
