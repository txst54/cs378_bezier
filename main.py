import numpy as np
import matplotlib.pyplot as plt
import random
from utils import bezier
import tqdm
import jax
import jax.numpy as jnp
import cma
from multiprocessing import Pool, cpu_count
from itertools import repeat
import gc

"""HYPERPARAMETERS"""
NUM_POINTS = 3  # n control points
P_DIM = 2  # n-dimensional bezier curve
IMSIZE = 12  # nxn image

INPUT_DIM = (IMSIZE * IMSIZE)
OUTPUT_DIM = (IMSIZE * IMSIZE)
PARAM_SIZE = INPUT_DIM * OUTPUT_DIM + OUTPUT_DIM
POP_SIZE = 128
INIT_STDEV = 0.2
NUM_SAMPLES = 100
TOTAL_GENS = 300
NUM_CORES = 10


class BezierSolver:
    def __init__(self):
        self.params = np.zeros((POP_SIZE, PARAM_SIZE))
        self.es = cma.CMAEvolutionStrategy(self.params[0],
                                           INIT_STDEV,
                                           {'popsize': POP_SIZE})

    @staticmethod
    def gen_train_data():
        # debug = []
        train_x = np.zeros((NUM_SAMPLES, IMSIZE * IMSIZE))
        train_y = np.zeros((NUM_SAMPLES, P_DIM * NUM_POINTS))
        for i in range(NUM_SAMPLES):
            num_points = NUM_POINTS  # Number of points, Dimension of points
            p_dim = P_DIM
            control_points = [[random.random() for __ in range(p_dim)] for ___ in range(num_points)]
            spline = bezier.eval_bezier(np.linspace(0, 1, IMSIZE), control_points)
            canvas = np.zeros((IMSIZE, IMSIZE))
            for j in range(len(spline)):
                y = min(int(spline[j][1] * IMSIZE), IMSIZE - 1)
                x = min(int(spline[j][0] * IMSIZE), IMSIZE - 1)
                canvas[y][x] = 1
            # debug.append(bezier)
            train_x[i] = np.ravel(np.array(canvas))
            train_y[i] = np.ravel(np.array(control_points))
        return train_x, train_y

    @staticmethod
    def logits_to_coords(logits):
        # Flatten the array and find the indices of the 3 highest values
        flattened_indices = np.argpartition(logits, -3)[-3:]
        # Convert the flattened indices back to 2D coordinates
        coords = np.ravel(np.array(np.unravel_index(flattened_indices, (IMSIZE, IMSIZE))).T) / IMSIZE
        return coords

    @staticmethod
    def mse(pred, y):
        return ((pred - y) ** 2).mean()

    @staticmethod
    def mlp_forward(params, obs):
        x = obs
        ss = 0
        ee = INPUT_DIM * OUTPUT_DIM
        w_in = params[ss:ee].reshape(INPUT_DIM, OUTPUT_DIM)
        ss = ee
        ee = ss + OUTPUT_DIM
        bias = params[ss:ee]
        x = jnp.tanh(jnp.dot(w_in, x) + bias)
        return x

    @staticmethod
    def eval_params(params):
        train_x, train_y = BezierSolver.gen_train_data()
        loss = 0
        for (x, y) in zip(train_x, train_y):
            logits = BezierSolver.mlp_forward(params, x)
            coords = BezierSolver.logits_to_coords(logits)
            error = BezierSolver.mse(coords, y)
            loss += error
        return loss

    def ask(self):
        self.params = np.array(self.es.ask())
        return self.params

    def tell(self, loss):
        loss = np.array(loss)
        self.es.tell(self.params, loss.tolist())

    def train(self):
        num_gens = TOTAL_GENS
        print("Training...")
        for gen in range(num_gens):
            self.ask()
            with Pool(NUM_CORES) as pool:
                loss = pool.map(BezierSolver.eval_params, self.params)
            self.tell(loss)
            print(f'gen={gen}, '
                  f'best_loss={np.min(loss)}, '
                  f'avg_loss={np.mean(loss)}, '
                  f'champion_idx={np.argmin(np.array(loss))}'
                  )
            if gen % 50 == 0:
                np.save(f'./params_{gen}.npy', self.params)
            del loss
            gc.collect()


if __name__ == '__main__':
    solver = BezierSolver()
    solver.train()
