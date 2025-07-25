import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import warnings
warnings.filterwarnings("error")


def relu(x):
    return np.maximum(0, x)


class EC:
    def __init__(self, hp):
        self.nec = hp['nec']
        self.si = hp['si']
        v1 = np.random.randn(self.nec)
        v1 /= np.linalg.norm(v1)
        v_orth = np.random.randn(self.nec)
        v_orth -= np.dot(v_orth, v1) * v1  # Remove the component along v1
        v_orth /= np.linalg.norm(v_orth)  # Normalize
        v2 = self.si * v1 + np.sqrt(1 - self.si ** 2) * v_orth
        print("EC:", np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
        self.A = v1.reshape(1, -1)  # Base vector in context A
        self.B = v2.reshape(1, -1)  # Base vector in context B
        self.h = np.zeros((1, self.nec))
        self.noise = hp['noise']

    def process(self, chamber):
        if chamber == 'A':  # EC activity in context A (at this time step)
            self.h = self.A + np.random.normal(0, self.noise, (1, self.nec))
        elif chamber == 'B':  # EC activity in context B (at this time step)
            self.h = self.B + np.random.normal(0, self.noise, (1, self.nec))
        else:
            print("EC Error")
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
        self.win_ori = np.random.normal(0, 1 / np.sqrt(self.ndg + 14 * self.add), (self.nec, self.ndg + 14 * self.add))

        self.win = np.copy(self.win_ori)  # EC-to-DG weights
        self.m = np.eye(self.ndg + 14 * self.add, self.ndg + 14 * self.add)  # DG recurrent weights
        self.b = np.zeros((1, self.ndg + 14 * self.add))  # bias term
        self.x = np.zeros((1, self.ndg + 14 * self.add))  # DGC membrane potentials

        # Feedforward learning rate, separated for EC-to-(baseline/newborn) DGCs
        self.lr_ff = np.full((self.nec, self.ndg + 14 * self.add), hp['lr_ff'])
        self.lr_ff[:, self.ndg:] = hp['lr_ff_new']

        # Bias term learning rate, separated for baseline/newborn DGCs
        self.lr_b = np.full((1, self.ndg + 14 * self.add), hp['lr_b'])
        self.lr_b[:, self.ndg:] = hp['lr_b_new']

        # Lateral learning rate, separated for all DGCs --> baseline/newborn DGCs
        self.lr_lat = np.full((self.ndg + 14 * self.add, self.ndg + 14 * self.add), hp['lr_lat'])
        self.lr_lat[:, self.ndg:] = hp['lr_lat_new']

        # Connectivity masks controlling the no. of DGCs available
        self.w_mask = np.zeros((self.nec, self.ndg + 14 * self.add))
        self.m_mask = np.zeros((self.ndg + 14 * self.add, self.ndg + 14 * self.add))

        self.hs = []

    def update(self, inputs, day):
        # Hebbian learning of EC-->DG, lateral weights, and bias term
        self.win = (1 - self.lr_ff) * self.win + np.transpose(inputs) @ self.h * self.lr_ff
        self.m = np.maximum((1 - self.lr_lat) * self.m + self.lr_lat * np.outer(self.h, self.h), 0)
        self.b = (1 - self.lr_b) * self.b + self.lr_b * self.h
        # Control the no. of DGCs
        self.win *= np.copy(self.w_mask)
        self.m *= np.copy(self.m_mask)
        # Avoid dividing 0
        np.fill_diagonal(self.m[(self.ndg + day * self.add):, (self.ndg + day * self.add):], 1)
        return

    def process(self, inputs):
        # Membrane potential updating, discretized with Euler method
        # Inputs from EC, lateral inhibition from other DGCs
        self.x += self.alpha * (-self.x + inputs @ self.win - self.b - self.h @ (self.m - np.diag(np.diag(self.m))))
        # Membrane potential --> firing rate
        # ReLU(x_i / M_ii), homeostatic plasticity (Pehlevan, ICASSP, 2019)
        self.h = relu((self.x - self.lbd1)/(self.lbd2 + np.diag(self.m).reshape(1, -1)))
        return self.h

    def reset(self):
        self.x = np.zeros((1, self.ndg + 14 * self.add))
        self.h = np.zeros((1, self.ndg + 14 * self.add))
        self.b = np.zeros((1, self.ndg + 14 * self.add))

    def turnon(self, day):
        # Adding new DGCs (no.=Delta N) by setting the corresponding entries in the connectivity matrices to 1
        if day > 0:
            self.win[:, (self.ndg + (day - 1) * self.add):] = np.copy(self.win_ori[:, (self.ndg + (day - 1) * self.add):])
        self.w_mask[:, :(self.ndg + day * self.add)] = 1
        self.win *= np.copy(self.w_mask)
        self.m_mask[:(self.ndg + day * self.add), :(self.ndg + day * self.add)] = 1
        self.m *= np.copy(self.m_mask)
        np.fill_diagonal(self.m[(self.ndg + day * self.add):, (self.ndg + day * self.add):], 1)


class Critic:
    def __init__(self, hp):
        self.ndg = hp['ndg']
        self.ncri = hp['ncri']
        self.tstep = hp['tstep']
        self.alpha = self.tstep / hp['tau']
        self.add = hp['add']
        self.win_ori = np.random.normal(0, 1/(self.ndg + 14 * self.add), (self.ndg + 14 * self.add, self.ncri))
        self.win = np.copy(self.win_ori)
        self.x = np.zeros((1, self.ncri))
        self.lr = hp['eta_cri']
        self.h = np.zeros((1, self.ncri))
        self.pastvalue = np.zeros((1, self.ncri))
        self.gamma = hp['gamma']

        self.w_mask = np.zeros((self.ndg + 14 * self.add, self.ncri))

    def process(self, inputs):
        self.pastvalue = np.copy(self.h)
        self.x += self.alpha * (-self.x + inputs @ self.win)
        self.h = np.copy(self.x)
        return self.h

    def compute_td(self, reward):
        # delta = reward + gamma * V - V_(t - 1)
        tderr = reward + self.gamma * self.h - self.pastvalue
        return tderr

    def update(self, inputs, tderr):
        dw = np.transpose(inputs) @ tderr
        self.win += self.lr * dw  # 2-factor rule
        self.win *= np.copy(self.w_mask)

    def reset(self):
        self.x = np.zeros((1, self.ncri))
        self.h = np.zeros((1, self.ncri))
        self.pastvalue = np.zeros((1, self.ncri))

    def turnon(self, day):
        if day > 0:
            self.win[(self.ndg + (day - 1) * self.add):, :] = np.copy(self.win_ori[(self.ndg + (day - 1) * self.add):, :])
        self.w_mask[:(self.ndg + day * self.add), :] = 1
        self.win *= np.copy(self.w_mask)


class Actor:
    def __init__(self, hp):
        self.ndg = hp['ndg']
        self.nact = hp['nact']
        self.tstep = hp['tstep']
        self.alpha = self.tstep / hp['tau']
        self.add = hp['add']
        self.win_ori = np.random.normal(0, 1/(self.ndg + 14 * self.add), (self.ndg + 14 * self.add, self.nact))
        self.win = np.copy(self.win_ori)
        self.x = np.zeros((1, self.nact))
        self.lr = hp['eta_act']
        self.h = np.zeros((1, self.nact))
        self.beta = hp['beta']
        self.onehotg = np.zeros(self.nact)

        self.w_mask = np.zeros((self.ndg + 14 * self.add, self.nact))

    def process(self, inputs):
        self.x += self.alpha * (-self.x + inputs @ self.win)
        self.h = np.copy(self.x)
        self.prob = np.exp(self.beta * self.h) / np.sum(np.exp(self.beta * self.h))
        # Softmax to model the lateral inhibition in the central amygdala
        A = np.random.choice(a=np.arange(self.nact), p=np.array(self.prob[0]))
        self.onehotg = np.zeros(self.nact)
        self.onehotg[A] = 1
        return self.onehotg

    def get_action(self):
        return self.onehotg

    def update(self, inputs, tderr, A):
        dw = np.transpose((A[:, None]) @ inputs)
        self.win += self.lr * tderr * dw  # 3-factor rule
        self.win *= np.copy(self.w_mask)

    def reset(self):
        self.x = np.zeros((1, self.nact))
        self.h = np.zeros((1, self.nact))

    def turnon(self, day):
        if day > 0:
            self.win[(self.ndg + (day - 1) * self.add):, :] = np.copy(self.win_ori[(self.ndg + (day - 1) * self.add):, :])
        self.w_mask[:(self.ndg + day * self.add), :] = 1
        self.win *= np.copy(self.w_mask)
