"""
###############   Parallel Trainer   ###############

# Simple Neat implementation in pytorch
# This is a Trader where it just looks into its past history nad the current position value and trades accordingly.
"""

import multiprocessing

import cv2
import numpy as np
import neat
from os import path
import pickle
from utils.data_utils import load_csv, split_data
from utils.reporter import LoggerReporter
from lib.env.TraderRenkoEnv import StockTradingEnv
from utils.cache import get_observations

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.recurrent_net import RecurrentNet

input_data_path = path.join('data', 'dataset', 'ADANIPORTS-EQ.csv')

feature_df = load_csv(input_data_path)

train_df, _ = split_data(feature_df)

params = {
    'look_back_window_size': 375*15,
    'enable_stationarization': True,
    'n_processes': multiprocessing.cpu_count(),
    'pre_computed_observation': False,
    'enable_env_logging': False,
    'observation_window': int(100)
}

max_env_steps = len(train_df) - 1

resume = False
restore_file = "neat-checkpoint-0"


def make_env(pre_obs, env_params):
    if env_params['pre_computed_observation']:
        return StockTradingEnv(train_df, pre_obs, **env_params)

    return StockTradingEnv(train_df, **env_params)


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return np.argmax(outputs, axis=1)


def run(n_generations, n_processes, pre_obs=None):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = path.join('data', 'config', 'config.cfg')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps, env_parms=params, env_pre_obs=pre_obs
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome_id, genome, config) for genome_id, genome in genomes)
            )
            for (genome_id, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:
        def eval_genomes(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(genome_id, genome, config)
                except Exception as e:
                    print(genome)
                    raise e

    if resume:
        pop = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        while True:
            pop = neat.Population(config)

            if 4 > len(pop.species.species) > 1:
                break

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(LoggerReporter(True))
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(1))

    winner = pop.run(eval_genomes, n_generations)

    # visualize.draw_net(config, winner)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    print(winner)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    if params['pre_computed_observation']:
        obs_dict = get_observations(train_df, params, True)
        
        # with open('obs_dict.pkl', 'wb') as output:
        #     pickle.dump(obs_dict, output, 1)

        run(n_generations=50, n_processes=2, pre_obs=obs_dict)
    else:
        run(n_generations=100, n_processes=2)
