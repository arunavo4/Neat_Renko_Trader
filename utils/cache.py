"""
    # @Arunavo Ray 08-07-19

    # Here we are trying to pre-compute the observations
    # The Ohlc and the Indicators.
    # Going through Log_Diff and Min_Max_Normalization

    # Default look_back window of 7 days
    # Multiprocessing the Observation calculation part
"""
import time
import multiprocessing as mp
from utils.data_utils import *
from lib.features.indicators import get_pattern_columns
from lib.features.transform import log_and_difference, max_min_normalize


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def _compute_next_obs(stationary_df, index_list, obs_dict, options):
    for current_idx in index_list:
        observations = stationary_df[-options['look_back_window_size'] + current_idx:current_idx].values
        observations = pd.DataFrame(observations, columns=stationary_df.columns)

        observations = observations[observations.columns.difference(['index', 'Date'])]

        if options['enable_stationarization']:
            scaled = log_and_difference(observations, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        else:
            scaled = observations

        scaled = max_min_normalize(scaled, columns=scaled.columns, pattern_columns=get_pattern_columns(),
                                   pattern_normalize=True)
        obs_dict[current_idx] = scaled.values[-1]


def get_observations(df, options, parallel=True):
    print("Pre_Computing_Observations! ")

    start = time.time()
    df = df.fillna(method='bfill').reset_index()
    stationary_df = df.copy()

    if int(options['look_back_window_size'] / 375) > 1:
        start_offset = int(375) * int(options['look_back_window_size'] / 375)
    else:
        start_offset = int(375)

    start_offset += 1
    max_env_steps = len(df) + 1

    indexes = [i for i in range(start_offset, max_env_steps)]

    chunks_list = chunks(indexes, options['n_processes'])

    if parallel:
        print("Parallel on {}".format(options['n_processes']))
        manager = mp.Manager()
        d = manager.dict()
        threads = []
        for i in range(options['n_processes']):
            threads.append(mp.Process(target=_compute_next_obs, args=(stationary_df, chunks_list[i], d, options,)))

        # start all threads
        for t in threads:
            t.start()

        # Join all threads
        for t in threads:
            t.join()
        # `d` is a DictProxy object that can be converted to dict
        end = time.time()
        print("Finished in parallel: {} Min".format(round((end-start)/60,2)))
        return dict(d)

    else:
        d = dict()

        _compute_next_obs(stationary_df, indexes, d, options)
        end = time.time()
        print("Finished in Single: {} Min".format(round((end-start)/60,2)))
        return d

