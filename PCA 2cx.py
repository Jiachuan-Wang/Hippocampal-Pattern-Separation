import sys
import os
sys.path.append(os.getcwd())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
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


def run_exp(reps, hp):
    days = 15
    tstep = hp['tstep']  # 20ms timestep
    step = int(1000/tstep)  # 1s/100ms
    ndg = hp['ndg']
    add = hp['add']
    for rep in range(reps):
        random.seed(rep)
        print('Replication {}'.format(rep))
        ec = EC(hp=hp)
        agent = HPC_Agent(hp=hp)
        for day in range(days):
            print('Trial {}'.format(day))
            dg_fr1 = np.zeros((180 * step, ndg + day * add))
            dg_fr2 = np.zeros((180 * step, ndg + day * add))
            agent.reset()
            agent.turnon(day)
            if day == 0:
                T_max = 180
            else:
                T_max = 180
                letters = ['A', 'B']
                for chamber in letters:
                    for i in range(T_max * step):
                        ec_act = ec.process(chamber)
                        agent.process(ec_act)
                        agent.learn(ec_act, day)
                        if chamber == 'A':
                            dg_fr1[i] = agent.dg.h
                        elif chamber == 'B':
                            dg_fr2[i] = agent.dg.h
                    agent.reset()
                all_dg = np.vstack((dg_fr1, dg_fr2))
                if day == 1:
                    all_dgs = all_dg
                else:
                    all_dgs = np.vstack((all_dgs, all_dg))
    return all_dgs


if __name__ == '__main__':
    hp = get_default_hp()
    reps = 1

    all_res = run_exp(reps, hp)
    all_res = all_res - all_res.mean(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(all_res)
    elbow = np.argmax(np.diff(np.cumsum(pca.explained_variance_ratio_)) < 0.01) + 1
    print('Elbow at PC:', elbow)

    day_index = np.repeat(np.arange(1, 14 + 1), 3600)
    context_labels = np.tile(np.concatenate([np.full(1800, 'A'), np.full(1800, 'B')]), 14)
    selected_days = [1, 7, 14]
    day_mask = np.isin(day_index, selected_days)

    # Downscale the image to prevent SVG from being too large
    def sample_mask(mask, day_index, selected_days, sample_size=200):
        final_mask = np.zeros_like(mask)
        for day in selected_days:
            day_indices = np.where(mask & (day_index == day))[0]
            if len(day_indices) > sample_size:
                sampled_indices = np.random.choice(day_indices, sample_size, replace=False)
            else:
                sampled_indices = day_indices
            final_mask[sampled_indices] = True
        return final_mask


    mask_A = context_labels == 'A'
    mask_B = context_labels == 'B'
    mask_A = sample_mask(mask_A & day_mask, day_index, selected_days)
    mask_B = sample_mask(mask_B & day_mask, day_index, selected_days)
    plt.figure(figsize=(3, 1.7), dpi=300)
    scatter_A = plt.scatter(X_pca[mask_A, 0], X_pca[mask_A, 1], c=day_index[mask_A], marker='o',
                            alpha=0.2, label='Context A', s=8)
    scatter_B = plt.scatter(X_pca[mask_B, 0], X_pca[mask_B, 1], c=day_index[mask_B], marker='^',
                            alpha=0.2, label='Context B', s=8)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=1, vmax=14)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Day', fontsize=8, labelpad=0.5)
    cbar.set_ticks(selected_days)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel('PC1', fontsize=8)
    plt.ylabel('PC2', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Context A', markeredgewidth=0,
               markerfacecolor='black', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='^', color='w', label='Context B', markeredgewidth=0,
               markerfacecolor='black', markersize=6, linestyle='None')
    ]
    plt.legend(handles=legend_elements, fontsize=8, loc='best', borderaxespad=0, labelspacing=0)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.axis('equal')
    plt.show()
