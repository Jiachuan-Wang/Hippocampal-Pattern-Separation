import sys
import os
sys.path.append(os.getcwd())


import numpy as np


def relu(x):
    return np.maximum(0, x)


class EC:
    def __init__(self, hp):
        self.nec = hp['nec']
        self.si = hp['si']
        self.ncx = 4

        if self.si <= -1 / (self.ncx - 1) or self.si >= 1:
            raise ValueError(f"Alpha must be in (-1/{self.ncx - 1}, 1) for n={self.ncx}")
        # Create Gram matrix
        G = np.full((self.ncx, self.ncx), self.si)
        np.fill_diagonal(G, 1.0)
        # Eigen-decompose
        eigvals, eigvecs = np.linalg.eigh(G)
        pos_idx = eigvals > 1e-10
        eigvals = eigvals[pos_idx]
        eigvecs = eigvecs[:, pos_idx]
        # Construct L with appropriate dimension
        L = eigvecs @ np.diag(np.sqrt(eigvals))  # shape: (n, rank)
        # Reduce to desired dimension
        if L.shape[1] < self.nec:
            # Pad with zeros to match target dimension
            padded = np.zeros((self.ncx, self.nec))
            padded[:, :L.shape[1]] = L
            L = padded
        elif L.shape[1] > self.nec:
            # Project down (best effort)
            L = L[:, :self.nec]
        # Normalize to unit vectors
        L /= np.linalg.norm(L, axis=1, keepdims=True)

        self.A = L[0].reshape(1, -1)
        self.B = L[1].reshape(1, -1)
        self.C = L[2].reshape(1, -1)
        self.D = L[3].reshape(1, -1)

        self.h = np.zeros((1, self.nec))
        self.noise = hp['noise']

    def process(self, chamber):
        if chamber == 'A':
            self.h = self.A + np.random.normal(0, self.noise, (1, self.nec))
        elif chamber == 'B':
            self.h = self.B + np.random.normal(0, self.noise, (1, self.nec))
        elif chamber == 'C':
            self.h = self.C + np.random.normal(0, self.noise, (1, self.nec))
        elif chamber == 'D':
            self.h = self.D + np.random.normal(0, self.noise, (1, self.nec))
        else:
            print("Error")
            return
        return self.h


class DG:
    def __init__(self, hp):
        self.nec = hp['nec']
        self.ndg = hp['ndg']
        self.tstep = hp['tstep']
        self.add = hp['add']
        self.alpha = self.tstep / hp['tau']
        self.lbd1 = hp['lbd1']
        self.lbd2 = hp['lbd2']

        self.h = np.zeros((1, self.ndg))
        self.win_ori = 0.1 * np.random.normal(0, 1, (self.nec, self.ndg + 14 * self.add))
        self.win_ori[:,self.ndg:] = np.random.normal(0, 1 / np.sqrt(self.ndg + 14 * self.add), (self.nec, 14 * self.add))

        self.win = np.copy(self.win_ori)
        self.m = np.eye(self.ndg + 14 * self.add, self.ndg + 14 * self.add)
        self.b = np.zeros((1, self.ndg + 14 * self.add))
        self.x = np.zeros((1, self.ndg + 14 * self.add))

        self.lr_ff = np.full((self.nec, self.ndg + 14 * self.add), hp['lr_ff'])
        self.lr_ff[:, self.ndg:] = hp['lr_ff_new']

        self.lr_b = np.full((1, self.ndg + 14 * self.add), hp['lr_b'])
        self.lr_b[:, self.ndg:] = hp['lr_b_new']

        self.lr_lat = np.full((self.ndg + 14 * self.add, self.ndg + 14 * self.add), hp['lr_lat'])
        self.lr_lat[:, self.ndg:] = hp['lr_lat_new']

        self.w_mask = np.zeros((self.nec, self.ndg + 14 * self.add))
        self.m_mask = np.zeros((self.ndg + 14 * self.add, self.ndg + 14 * self.add))

        self.hs = []

    def update(self, inputs, day):
        self.win = (1 - self.lr_ff) * self.win + np.transpose(inputs) @ self.h * self.lr_ff
        self.m = np.maximum((1 - self.lr_lat) * self.m + self.lr_lat * np.outer(self.h, self.h), 0)
        self.b = (1 - self.lr_b) * self.b + self.lr_b * self.h
        np.fill_diagonal(self.m[(self.ndg + day * self.add):, (self.ndg + day * self.add):], 1)
        return

    def process(self, inputs):
        self.x += self.alpha * (-self.x + inputs @ self.win - self.b - self.h @ (self.m - np.diag(np.diag(self.m))))
        self.h = relu((self.x - self.lbd1)/(self.lbd2 + np.diag(self.m).reshape(1, -1)))
        return self.h

    def reset(self):
        self.x = np.zeros((1, self.ndg + 14 * self.add))
        self.h = np.zeros((1, self.ndg + 14 * self.add))
        self.b = np.zeros((1, self.ndg + 14 * self.add))

    def turnon(self, day):
        if day > 0:
            self.win[:, (self.ndg + (day - 1) * self.add):] = np.copy(self.win_ori[:, (self.ndg + (day - 1) * self.add):])
        self.w_mask[:, :(self.ndg + day * self.add)] = 1
        self.win *= np.copy(self.w_mask)
        self.m_mask[:(self.ndg + day * self.add), :(self.ndg + day * self.add)] = 1
        self.m *= np.copy(self.m_mask)
        np.fill_diagonal(self.m[(self.ndg + day * self.add):, (self.ndg + day * self.add):], 1)


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


def run_exp(reps, hp):
    days = 15
    tstep = hp['tstep']  # 20ms timestep
    step = int(1000 / tstep)  # 1s/100ms
    ndg = hp['ndg']
    add = hp['add']
    cos_simis = np.array(())
    ec = EC(hp=hp)
    agent = HPC_Agent(hp=hp)
    for day in range(days):
        dg_fr1 = np.zeros((180 * step, ndg + 14 * add))
        dg_fr2 = np.zeros((180 * step, ndg + 14 * add))
        dg_fr3 = np.zeros((180 * step, ndg + 14 * add))
        dg_fr4 = np.zeros((180 * step, ndg + 14 * add))
        agent.reset()
        agent.turnon(day)
        print('Trial {}'.format(day))
        if day == 0:
            #for i in range(546 * step):
            ec_act = ec.process('A')
                #agent.process(ec_act)
                #agent.learn(ec_act, day)
        else:
            T_max = 180
            letters = ['A', 'B', 'C', 'D']
            #random.shuffle(letters)
            for chamber in letters:
                for i in range(T_max * step):
                    ec_act = ec.process(chamber)
                    agent.process(ec_act)
                    agent.learn(ec_act, day)
                    if chamber == 'A':
                        dg_fr1[i] = agent.dg.h
                    elif chamber == 'B':
                        dg_fr2[i] = agent.dg.h
                    elif chamber == 'C':
                        dg_fr3[i] = agent.dg.h
                    elif chamber == 'D':
                        dg_fr4[i] = agent.dg.h
                agent.reset()
            a = dg_fr1[:, :(ndg + day * add)]
            b = dg_fr2[:, :(ndg + day * add)]
            c = dg_fr3[:, :(ndg + day * add)]
            d = dg_fr4[:, :(ndg + day * add)]
            all_dg = np.vstack((a, b, c, d))
            if day == 1:
                all_dgs = all_dg
            else:
                all_dgs = np.vstack((all_dgs, all_dg))
    return all_dgs


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
    reps = 1
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NEW
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D


    all_res = run_exp(reps, hp)
    all_res = all_res - all_res.mean(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(all_res)
    elbow = np.argmax(np.diff(np.cumsum(pca.explained_variance_ratio_)) < 0.01) + 1
    print('Elbow at PC:', elbow)
    day_index = np.repeat(np.arange(1, 14 + 1), 7200)
    context_labels = np.tile(np.concatenate([np.full(1800, 'A'), np.full(1800, 'B'),
                                             np.full(1800, 'C'), np.full(1800, 'D')]), 14)
    mask_A = context_labels == 'A'
    mask_B = context_labels == 'B'
    mask_C = context_labels == 'C'
    mask_D = context_labels == 'D'
    selected_days = [1, 7, 14]
    day_mask = np.isin(day_index, selected_days)
    mask_A = mask_A & day_mask
    mask_B = mask_B & day_mask
    mask_C = mask_C & day_mask
    mask_D = mask_D & day_mask

    def random_subset(mask, n=200):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = np.random.choice(idx, size=n, replace=False)
        return idx

    mask_A = random_subset(mask_A)
    mask_B = random_subset(mask_B)
    mask_C = random_subset(mask_C)
    mask_D = random_subset(mask_D)

    # dpi = 100  # Dots per inch
    fig = plt.figure(figsize=(4.2, 3), dpi=300)
    ax = fig.add_subplot(111, projection='3d')  # NEW
    norm = Normalize(vmin=1, vmax=14)
    colors = norm(day_index)
    # Plot each context in 3D
    scatter_A = ax.scatter(X_pca[mask_A, 0], X_pca[mask_A, 1], X_pca[mask_A, 2],
                           c=day_index[mask_A], cmap='viridis', marker='o',
                           alpha=0.1, label='Context A', s=8)
    scatter_B = ax.scatter(X_pca[mask_B, 0], X_pca[mask_B, 1], X_pca[mask_B, 2],
                           c=day_index[mask_B], cmap='viridis', marker='^',
                           alpha=0.1, label='Context B', s=8)
    scatter_C = ax.scatter(X_pca[mask_C, 0], X_pca[mask_C, 1], X_pca[mask_C, 2],
                           c=day_index[mask_C], cmap='viridis', marker='s',
                           alpha=0.1, label='Context C', s=8)
    scatter_D = ax.scatter(X_pca[mask_D, 0], X_pca[mask_D, 1], X_pca[mask_D, 2],
                           c=day_index[mask_D], cmap='viridis', marker='*',
                           alpha=0.1, label='Context D', s=8)
    # Colorbar
    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Day', fontsize=8, labelpad=0.5)
    cbar.set_ticks(selected_days)
    cbar.ax.tick_params(labelsize=8)
    # Axis labels
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_zlabel('PC3', fontsize=8)
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', label='Context A', markerfacecolor='none', markersize=6,
               linestyle='None'),
        Line2D([0], [0], marker='^', label='Context B', markerfacecolor='none', markersize=6,
               linestyle='None'),
        Line2D([0], [0], marker='s', label='Context C', markerfacecolor='none', markersize=6,
               linestyle='None'),
        Line2D([0], [0], marker='*', label='Context D', markerfacecolor='none', markersize=6,
               linestyle='None')
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=8,
        loc='center left',
        bbox_to_anchor=(-0.4, 0.5),  # moves it leftward
        frameon=False
    )
    #plt.tight_layout()
    plt.show()
