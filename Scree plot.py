import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import random
import multiprocessing as mp
from sklearn.decomposition import PCA
from model import *


class HPC_Agent:
    def __init__(self, hp):
        self.hp = hp
        self.dg = DG(hp)

    def process(self, ec_act):
        dg_act = self.dg.process(ec_act)
        return dg_act

    def learn(self, ec_act, day):
        self.dg.update(ec_act, day)
        return

    def reset(self):
        self.dg.reset()

    def turnon(self, day):
        self.dg.turnon(day)


def run_single_rep(rep, hp):
    days = 15
    tstep = hp['tstep']
    step = int(1000 / tstep)
    ndg = hp['ndg']
    add = hp['add']
    print(rep)
    random.seed(rep)
    ec = EC(hp=hp)
    agent = HPC_Agent(hp=hp)
    explained_variance = {}
    for day in range(days):
        dg_fr1 = np.zeros((180 * step, ndg + day * add))
        dg_fr2 = np.zeros((180 * step, ndg + day * add))
        agent.reset()
        agent.turnon(day)
        if day == 0:
            """
            for i in range(546 * step):
                ec_act = ec.process('A')
                agent.process(ec_act)
                agent.learn(ec_act, day)
            """
            print('No pre-training')
        else:
            T_max = 180
            letters = ['A', 'B']
            random.shuffle(letters)
            for chamber in letters:
                for i in range(T_max * step):
                    ec_act = ec.process(chamber)
                    agent.process(ec_act)
                    agent.learn(ec_act, day)
                    if chamber == 'A':
                        dg_fr1[i] = agent.dg.h[:, :(ndg + day * add)]
                    elif chamber == 'B':
                        dg_fr2[i] = agent.dg.h[:, :(ndg + day * add)]
                agent.reset()
        if day in [1, 7, 14]:
            all_dg = np.vstack((dg_fr1, dg_fr2))
            pca = PCA()
            A_scaled = all_dg - all_dg.mean(axis=0)
            pca.fit(A_scaled)
            explained_variance[day] = np.cumsum(pca.explained_variance_ratio_[:100])
    df = pd.DataFrame(
        [values for values in explained_variance.values()],
        index=explained_variance.keys(),
        columns=[f'PC {i + 1}' for i in range(100)]
    )
    # Reset index and rename the index column to 'days'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Day'}, inplace=True)
    res = pd.DataFrame(df)
    return res


def run_exp_parallel(reps, hp, n_workers=None):
    """ Runs the experiment in parallel across multiple cores. """
    n_workers = n_workers or mp.cpu_count()  # Use all available cores by default
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(run_single_rep, [(rep, hp) for rep in range(reps)])
    # Combine results from all replications
    all_res = pd.concat([res for res in results], ignore_index=False)
    return all_res


def get_default_hp():
    hp = {
        'tstep': 100,  # time step in ms
        'tau': 2000,  # time constant
        'nec': 8,  # EC size
        'ndg': 100,  # DG size
        'add': 0,  # neurogenesis rate (Delta N)
        'lr_ff': 2.5e-5,  # EC-to-DG (feedforward) learning rate for the fully developed (baseline) DGCs
        'lr_lat': 2.5e-5,  # lateral learning rate for the fully developed (baseline) DGCs
        'lr_b': 2.5e-5,  # learning rate for bias term for the fully developed (baseline) DGCs
        'si': 0.1,  # EC inputs similarity
        'lbd1': 0,  # regularization factors
        'lbd2': 0,
        'noise': 1 / 2  # Gaussian noise magnitude for EC input
    }
    return hp


if __name__ == '__main__':
    hp = get_default_hp()
    reps = 50

    all_res = run_exp_parallel(reps, hp)
    all_res.to_csv('PCs_{}lr_{}rep.csv'.format(hp['lr_ff'], reps))
