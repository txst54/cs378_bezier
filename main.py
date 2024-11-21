import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Sequence, Tuple
import random
import time
import h5py
from utils import bezier
import tqdm
import jax
import jax.numpy as jnp
import cma
from multiprocessing import Pool, cpu_count
import gc

"""HYPERPARAMETERS"""
NUM_POINTS = 3  # n control points
P_DIM = 2  # n-dimensional bezier curve
IMSIZE = 12  # nxn image

INPUT_DIM = (IMSIZE * IMSIZE)
OUTPUT_DIM = (IMSIZE * IMSIZE)
PARAM_SIZE = INPUT_DIM * OUTPUT_DIM + OUTPUT_DIM
POP_SIZE = 4
INIT_STDEV = 0.2
NUM_SAMPLES = 100
TOTAL_GENS = 300
NUM_CORES = 10


class BezierSolver:
    def __init__(self):
        self.params = np.zeros((POP_SIZE, PARAM_SIZE))
        self.debug = []
        self.train_x = []
        self.train_y = []
        self.es = cma.CMAEvolutionStrategy(self.params[0],
                                           INIT_STDEV,
                                           {'popsize': POP_SIZE})

    def gen_train_data(self):
        for _ in tqdm.tqdm(range(NUM_SAMPLES)):
            num_points = NUM_POINTS  # Number of points, Dimension of points
            p_dim = P_DIM
            x = [[random.random for __ in range(p_dim)] for ___ in range(num_points)]
            spline = bezier.eval_bezier(np.linspace(0, 1, IMSIZE), x)
            canvas = np.zeros((IMSIZE, IMSIZE))
            for j in range(len(spline)):
                y = min(int(spline[j][1] * IMSIZE), IMSIZE - 1)
                x = min(int(spline[j][0] * IMSIZE), IMSIZE - 1)
                canvas[y][x] = 1
            self.debug.append(bezier)
            self.train_x.append(canvas)
            self.train_y.append(x)

    def logits_to_coords(self, logits):
        # Flatten the array and find the indices of the 3 highest values
        flattened_indices = np.argpartition(logits, -3)[-3:]
        # Convert the flattened indices back to 2D coordinates
        coords = np.array(np.unravel_index(flattened_indices, (IMSIZE, IMSIZE))).T
        return coords

    def mse(self, pred, y):
        return ((pred - y) ** 2).mean()

    def mlp_forward(self, params, obs):
        x = obs
        ss = 0
        ee = INPUT_DIM * OUTPUT_DIM
        w_in = params[ss:ee].reshape(INPUT_DIM, OUTPUT_DIM)
        ss = ee
        ee = ss + OUTPUT_DIM
        bias = params[ss:ee]
        x = jnp.tanh(jnp.dot(w_in, x) + bias)
        return x

    def ask(self):
        self.params = np.array(self.es.ask())
        return self.params

    def tell(self, loss):
        loss = np.array(loss)
        self.es.tell(self.params, loss.tolist())

    def eval_params(self, params):
        print("Evaluating...")
        loss = 0
        for (x, y) in zip(self.train_x, self.train_y):
            logits = self.mlp_forward(params, x)
            coords = self.logits_to_coords(logits)
            error = self.mse(coords, y)
            loss += error
        return loss

    def train(self):
        num_gens = TOTAL_GENS
        print("Training...")
        for gen in range(num_gens):
            with Pool(NUM_CORES) as pool:
                self.ask()
                loss = pool.map(self.eval_params, self.params)
            print(f'Champion Index: {np.argmin(np.array(loss))}')
            self.tell(loss)
            print(f'gen={gen}, best_loss={np.min(loss)}, avg_loss={np.mean(loss)}')
            if gen % 50 == 0:
                np.save(f'./params_{gen}.npy', self.params)
            del loss
            gc.collect()


if __name__ == '__main__':
    solver = BezierSolver()
    solver.train()
