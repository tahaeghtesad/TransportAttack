import multiprocessing
import os

import numpy as np
import tqdm
from tbparse import SummaryReader


# this function should return a list of all directories in 'path' folder.
def list_runs(path):
    return os.listdir(path)


def get_run_reward(run):
    reader = SummaryReader(f'{run}')
    df = reader.scalars
    return df.loc[df['tag'] == 'env/original_reward'].value[:-100].mean()


def get_top_runs(rewards):
    return np.argpartition(rewards, -10)[-10:]


if __name__ == '__main__':

    path = 'logs'
    runs = list_runs(path)
    runs = [os.path.join(path, run) for run in runs]

    with multiprocessing.Pool(1) as pool:
        rewards = list(tqdm.tqdm(pool.imap(get_run_reward, runs), total=len(runs)))

    for indices in get_top_runs(rewards):
        print(runs[indices], rewards[indices])