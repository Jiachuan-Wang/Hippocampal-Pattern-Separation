import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
import random
import multiprocessing as mp
import warnings

warnings.filterwarnings("error")

from model import *


class Full_Agent:
    def __init__(self, hp):
        self.dg = DG(hp)
        self.actor = Actor(hp)
        self.critic = Critic(hp)

    def process(self, ec_act):
        dg_act = self.dg.process(ec_act)
        self.critic.process(dg_act)
        act_act = self.actor.process(dg_act)
        return act_act

    def learn(self, reward, ec_act, day):
        tderr = self.critic.compute_td(reward)
        self.dg.update(ec_act, day)
        self.critic.update(self.dg.h, tderr)
        self.actor.update(self.dg.h, tderr, self.actor.onehotg)
        return

    def reset(self):
        self.dg.reset()
        self.actor.reset()
        self.critic.reset()

    def turnon(self, day):
        self.dg.turnon(day)
        self.actor.turnon(day)
        self.critic.turnon(day)


def get_reward(shock, freeze, sig, r, eps):
    gen_rand = np.random.uniform(0, 1, 1)
    reward = 0
    if gen_rand < eps and (freeze == 0):
        reward = r  # If moving, getting a small reward with a small probability
        # No reward in the main simulations
    if shock and (freeze == 0):  # freeze [1, 0]
        reward = -sig  # If moving during the foot shock period, receiving strong punishment
    return reward


def experiment(day, sub):
    if day == 0:
        if sub == 'preshock':
            T_max = 182
        elif sub == 'after1':
            T_max = 122
        elif sub == 'after2':
            T_max = 122
        elif sub == 'after3':
            T_max = 120
        chamber = None
    elif day in [1, 3, 5, 6, 9, 11, 12]:  # Agents are placed in chamber A first, then chamber B (as in the experiment)
        if sub == 'first':
            T_max = 182
            chamber = 'A'
        else:
            T_max = 180
            chamber = 'B'
    else:  # Agents are placed in chamber B first
        if sub == 'first':
            T_max = 180
            chamber = 'B'
        else:
            T_max = 182
            chamber = 'A'
    return T_max, chamber


def run_single_rep(hp, rep, seed):
    np.random.seed(seed)
    days = 15
    sig = hp['punish']
    r = hp['reward']
    eps = hp['eps']
    tstep = hp['tstep']
    step = int(1000 / tstep)  # 1s/'time step' ms
    ndg = hp['ndg']
    add = hp['add']
    all_res = pd.DataFrame()
    print('Replication {}'.format(rep))
    similarities = np.array(())
    A = np.array(())
    B = np.array(())
    random.seed(rep)
    ec = EC(hp=hp)
    agent = Full_Agent(hp=hp)
    for day in range(days):
        dg_fr1 = np.zeros((180 * step, ndg + day * add))
        dg_fr2 = np.zeros((180 * step, ndg + day * add))
        agent.reset()
        agent.turnon(day)
        #  Day 0: 3 stages with recording and foot shocks; Day 1-14: record for 180 s and foot shock of 2 s.
        if day == 0:
            T_max, cue = experiment(day, 'preshock')
            cnt = 0
            for i in range(T_max * step):
                ec_act = ec.process('A')
                action = agent.process(ec_act)
                if i in range(step * 180, 181 * step):
                    shock = True
                else:
                    shock = False
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0 and i < 180 * step:
                    cnt += 1
                agent.learn(reward, ec_act, day)
            # print('Trial {} Preshock | freezing ratio {}'.format(day, cnt / (180 * step)))
            A = np.append(A, cnt / (180 * step))
            B = np.append(B, np.nan)
            T_max, cue = experiment(day, 'after1')
            cnt = 0
            for i in range(T_max * step):
                ec_act = ec.process('A')
                action = agent.process(ec_act)
                if i in range(step * 120, 121 * step):
                    shock = True
                else:
                    shock = False
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0 and i < 120 * step:
                    cnt += 1
                agent.learn(reward, ec_act, day)
            # print('Trial {} After Shock 1 | freezing ratio {}'.format(day, cnt / (120 * step)))
            A = np.append(A, cnt / (120 * step))
            B = np.append(B, np.nan)
            T_max, cue = experiment(day, 'after2')
            cnt = 0
            for i in range(T_max * step):
                ec_act = ec.process('A')
                action = agent.process(ec_act)
                if i in range(step * 120, 121 * step):
                    shock = True
                else:
                    shock = False
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0 and i < 120 * step:
                    cnt += 1
                agent.learn(reward, ec_act, day)
            # print('Trial {} After Shock 2 | freezing ratio {}'.format(day, cnt / (120 * step)))
            A = np.append(A, cnt / (120 * step))
            B = np.append(B, np.nan)
            T_max, cue = experiment(day, 'after3')
            cnt = 0
            for i in range(T_max * step):
                ec_act = ec.process('A')
                action = agent.process(ec_act)
                shock = False
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0:
                    cnt += 1
                agent.learn(reward, ec_act, day)
            # print('Trial {} After Shock 3 | freezing ratio {}'.format(day, cnt / (120 * step)))
            A = np.append(A, cnt / (120 * step))
            B = np.append(B, np.nan)
        else:
            T_max, cue = experiment(day, 'first')
            cnt = 0
            for i in range(T_max * step):
                ec_act = ec.process(cue)
                action = agent.process(ec_act)
                shock = False
                if T_max > 180 and (i in range(step * 180, 182 * step)):
                    shock = True
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0 and i < 180 * step:
                    cnt += 1
                agent.learn(reward, ec_act, day)
                if cue == 'A' and i < 180 * step:
                    dg_fr1[i] = agent.dg.h[:, :(ndg + day * add)]
                elif cue == 'B':
                    dg_fr2[i] = agent.dg.h[:, :(ndg + day * add)]
            # print('Trial {} {} | freezing ratio {}'.format(day, cue, cnt / (180 * step)))
            if cue == 'A':
                A = np.append(A, cnt / (180 * step))
            else:
                B = np.append(B, cnt / (180 * step))
            # last_T_max = T_max
            T_max, cue = experiment(day, 'second')
            cnt = 0
            agent.reset()
            for i in range(T_max * step):
                ec_act = ec.process(cue)
                action = agent.process(ec_act)
                shock = False
                if T_max > 180 and (i in range(step * 180, 182 * step)):
                    shock = True
                reward = get_reward(shock, action[0], sig, r, eps)
                if action[0] == 1.0 and i < 180 * step:
                    cnt += 1
                agent.learn(reward, ec_act, day)
                if cue == 'A' and i < 180 * step:
                    dg_fr1[i] = agent.dg.h[:, :(ndg + day * add)]
                elif cue == 'B':
                    dg_fr2[i] = agent.dg.h[:, :(ndg + day * add)]
            # print('Trial {} {} | freezing ratio {}'.format(day, cue, cnt / (180 * step)))
            cos_simi = np.trace(np.transpose(dg_fr1) @ dg_fr2) / (
                    np.linalg.norm(dg_fr1, 'fro') * np.linalg.norm(dg_fr2, 'fro'))
            similarities = np.append(similarities, cos_simi)
            if cue == 'A':
                A = np.append(A, cnt / (180 * step))
            else:
                B = np.append(B, cnt / (180 * step))
    res = pd.DataFrame({'A': A, 'B': B})  # Freezing ratio (behavioral) results
    # print(res)
    cos = pd.DataFrame({'Cosine': similarities})  # Cosine similarity results
    print(cos)
    return res, cos

    
def run_exp_parallel(reps, hp, n_workers=None):
    """ Runs the experiment in parallel across multiple cores. """
     n_workers = n_workers or mp.cpu_count()  # Use all available cores by default
    call_id = next(_call_counter)
    seeds = [42 + call_id * 10_000 + rep for rep in range(reps)]
    args = [(hp, rep, seeds[rep]) for rep in range(reps)]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(run_single_rep, args)
    # Combine results from all replications
    all_res = pd.concat([res for res, _ in results], ignore_index=False)  # Freezing ratio (behavioral) results
    coss = pd.concat([cos for _, cos in results], ignore_index=False)  # Cosine similarity results
    return all_res, coss


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
        'nact': 2,  # Actor size
        'ncri': 1,  # Critic size
        'eta_cri': 5e-2,  # DG-to-critic learning rate
        'eta_act': 2.5e-2,  # DG-to-actor learning rate
        'beta': 1,  # exploitation factor
        'punish': 10,  # punishment amplitude
        'reward': 0,  # reward amplitude (no reward)
        'eps': 0.01,  # reward probability (not used)
        'gamma': 0.95,  # discount factor
        'noise': 1 / 2  # Gaussian noise magnitude for EC input
    }
    return hp


if __name__ == '__main__':
    import itertools
    _call_counter = itertools.count()
    hp = get_default_hp()
    reps = 500  # number of replications / agents

    for lr in [1e-4, 1.5e-5, 1e-5, 1e-6, 1e-7, 2.5e-5, 5e-5, 5e-6, 7.5e-5]:
        hp['add'] = 1
        hp['lr_ff'] = lr
        hp['lr_ff_new'] = lr
        hp['lr_lat'] = lr
        hp['lr_lat_new'] = lr
        hp['lr_b'] = lr
        hp['lr_b_new'] = lr
        all_res = run_exp_parallel(reps, hp)
        # Neurogenesis with mature DGCs
        all_res[0].to_csv('FreezingRes_matureNG_{}lr_{}rep.csv'.format(lr, reps))
        all_res[1].to_csv('CosSimRes_matureNG_{}lr_{}rep.csv'.format(lr, reps))

        hp = get_default_hp()
        hp['add'] = 0
        hp['lr_ff'] = lr
        hp['lr_ff_new'] = lr
        hp['lr_lat'] = lr
        hp['lr_lat_new'] = lr
        hp['lr_b'] = lr
        hp['lr_b_new'] = lr
        all_res = run_exp_parallel(reps, hp)
        # No neurogenesis
        all_res[0].to_csv('FreezingRes_noNG_{}lr_{}rep.csv'.format(lr, reps))
        all_res[1].to_csv('CosSimRes_noNG_{}lr_{}rep.csv'.format(lr, reps))

        hp = get_default_hp()
        hp['add'] = 1
        hp['lr_ff'] = lr
        hp['lr_ff_new'] = lr * 2
        hp['lr_lat'] = lr
        hp['lr_lat_new'] = lr / 2
        hp['lr_b'] = lr
        hp['lr_b_new'] = lr / 2
        all_res = run_exp_parallel(reps, hp)
        # # Neurogenesis with young DGCs
        all_res[0].to_csv('FreezingRes_youngNG_{}lr_{}rep.csv'.format(lr, reps))
        all_res[1].to_csv('CosSimRes_youngNG_{}lr_{}rep.csv'.format(lr, reps))


