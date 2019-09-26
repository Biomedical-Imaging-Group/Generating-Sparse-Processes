import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class L_spline():

    def __init__(self, L, w):
        self.L = L
        self.w = w

        self.lmda = None
        self.impulses = None

    def set_lambda(self, lmda):
        self.lmda = lmda

    def sample(self, T):
        self.impulses = self.w.impulse_approx(self.lmda, T)
    
    def eval(self, t):
        val_at_t = 0
        for knot, jump in self.impulses:
            val_at_t += jump*self.L.green(t-knot)
        return val_at_t

    def green_grid_samples(self, T, step, show=True, ax=None):

        impulses = self.impulses

        grid = np.arange(0, T, step)
        s = np.zeros_like(grid)
        for i in (range(len(grid))):
            t = grid[i]
            for knot, jump in impulses:
                s[i] += jump*self.L.green(t - knot)

        if show:
            ax = ax or plt.gca()
            ax.plot(grid, s)
        return s

    def get_increment_process(self, T, step):
        self.L.set_discretization_step(step)

        impulses = self.impulses

        grid = np.arange(0, T+step, step)
        u = 0.00*np.zeros_like(grid)
        for i in (range(len(impulses))):
            knot, jump  = impulses[i]
            main_grid_point = int(np.floor(knot/step))
            for grid_point in range(max(0, main_grid_point), min(len(grid), main_grid_point + self.L.degree+1)):
                t = grid[grid_point]
                u[grid_point] += jump * self.L.b_spline(t - knot)

        return u

    def get_grid_samples(self, T, step, show=True, ax=None):

        u = self.get_increment_process(T, step)
        s = self.L.discrete_inverse(u)
        
        grid = np.arange(0, T+step, step)

        if show:
            ax = ax or plt.gca()
            ax.step(grid, s, where='post', lw = 1.5)
        return s
