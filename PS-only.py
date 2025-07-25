import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
import random
import multiprocessing as mp
import warnings

warnings.filterwarnings("error")

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
    random.seed(rep)
    cos_simis = np.array(())
    ec = EC(hp=hp)
    agent = HPC_Agent(hp=hp)
    for day in range(days):
        dg_fr1 = np.zeros((180 * step, ndg + day * add))
        dg_fr2 = np.zeros((180 * step, ndg + day * add))
        agent.reset()
        agent.turnon(day)
        if day == 0:  # No learning, just for testing
            for i in range(180 * step):
                ec_act = ec.process('A')
                agent.process(ec_act)
                dg_fr1[i] = agent.dg.h[:, :(ndg + day * add)]
            agent.reset()
            for i in range(180 * step):
                ec_act = ec.process('B')
                agent.process(ec_act)
                dg_fr2[i] = agent.dg.h[:, :(ndg + day * add)]
            cos_simi = np.trace(np.transpose(dg_fr1) @ dg_fr2) \
                       / (np.linalg.norm(dg_fr1, 'fro') * np.linalg.norm(dg_fr2, 'fro'))
            cos_simis = np.append(cos_simis, cos_simi)
        else:
            T_max = 180
            letters = ['A', 'B']
            random.shuffle(letters)
            for chamber in letters:  # Placed in each chamber, random order
                for i in range(T_max * step):
                    ec_act = ec.process(chamber)
                    agent.process(ec_act)
                    agent.learn(ec_act, day)
                    if chamber == 'A':
                        dg_fr1[i] = agent.dg.h[:, :(ndg + day * add)]
                    elif chamber == 'B':
                        dg_fr2[i] = agent.dg.h[:, :(ndg + day * add)]
                agent.reset()
            # Calculate the population cosine similarity using the DG activity in A vs. B
            # Only use the 'activated' DGCs
            cos_simi = np.trace(np.transpose(dg_fr1) @ dg_fr2) \
                       / (np.linalg.norm(dg_fr1, 'fro') * np.linalg.norm(dg_fr2, 'fro'))
            # print(np.linalg.norm(dg_fr1), np.linalg.norm(dg_fr2))
            cos_simis = np.append(cos_simis, cos_simi)
            print(day, cos_simi)
    res = pd.DataFrame({'Cosine': cos_simis})
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
        'lr_ff_new': 2.5e-5,  # EC-to-DG (feedforward) learning rate for the newborn DGCs
        'lr_lat_new': 2.5e-5,  # lateral learning rate for the newborn DGCs
        'lr_b_new': 2.5e-5,  # learning rate for bias term for the newborn DGCs
        'si': 0.1,  # EC inputs similarity
        'lbd1': 0,  # regularization factors
        'lbd2': 0,
        'noise': 1 / 2  # Gaussian noise magnitude for EC input
    }
    return hp


if __name__ == '__main__':
    hp = get_default_hp()
    reps = 30

    for si in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        hp['si'] = si
        all_res = run_exp_parallel(reps, hp)
        all_res.to_csv('CosSimRes_{}si_{}rep.csv'.format(si, reps))
