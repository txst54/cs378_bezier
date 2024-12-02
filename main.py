import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from utils import bezier
import jax
import jax.numpy as jnp
import cma
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory
import gc

import psutil
import os


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


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
NUM_CORES = 1


class BezierSolver:
    def __init__(self, params=None):
        print(f'Current memory usage: {memory_usage()}')
        if params is None:
            params = np.zeros((POP_SIZE, PARAM_SIZE), dtype=np.float32)
        self.params_shm = shared_memory.SharedMemory(create=True, size=params.nbytes)
        self.set_params(params)
        print(f'Current memory usage: {memory_usage()}')
        self.es = cma.CMAEvolutionStrategy(self.params[0],
                                           INIT_STDEV,
                                           {'popsize': POP_SIZE})
        print(f'Current memory usage: {memory_usage()}')
        train_x, train_y = self.gen_train_data()
        self.train_x_shm = shared_memory.SharedMemory(create=True, size=train_x.nbytes)
        self.train_y_shm = shared_memory.SharedMemory(create=True, size=train_y.nbytes)
        self.train_x_shape = train_x.shape
        self.train_y_shape = train_y.shape
        print(f'Current memory usage: {memory_usage()}')

        # Copy training data to shared memory
        np.ndarray(self.train_x_shape, dtype=train_x.dtype, buffer=self.train_x_shm.buf)[:] = train_x
        np.ndarray(self.train_y_shape, dtype=train_y.dtype, buffer=self.train_y_shm.buf)[:] = train_y

    def set_params(self, val):
        np.ndarray((POP_SIZE, PARAM_SIZE), dtype=np.float32, buffer=self.params_shm.buf)[:] = val

    @staticmethod
    def gen_train_data():
        # debug = []
        train_x = np.zeros((NUM_SAMPLES, IMSIZE * IMSIZE), dtype=np.float32)
        train_y = np.zeros((NUM_SAMPLES, P_DIM * NUM_POINTS), dtype=np.float32)
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
    def eval_params(params_chunk, train_x_shm_name, train_y_shm_name, train_x_shape, train_y_shape):
        # Attach to shared memory
        train_x_shm = shared_memory.SharedMemory(name=train_x_shm_name)
        train_y_shm = shared_memory.SharedMemory(name=train_y_shm_name)

        # Now use np.ndarray to access the shared memory block
        train_x = np.ndarray(train_x_shape, dtype=np.float32, buffer=train_x_shm.buf)
        train_y = np.ndarray(train_y_shape, dtype=np.float32, buffer=train_y_shm.buf)
        # Initialize an array to store losses for this chunk
        losses = []

        # Evaluate each parameter set in the chunk
        for params in params_chunk:
            loss = 0
            print(f'Current memory usage: {memory_usage()}')
            for x, y in zip(train_x, train_y):  # Iterate over training samples
                logits = BezierSolver.mlp_forward(params, x)
                coords = BezierSolver.logits_to_coords(logits)
                error = BezierSolver.mse(coords, y)
                loss += error
            losses.append(loss)  # Append the total loss for this parameter set
        return losses

    @staticmethod
    def plot_heatmap(x, y, logits_heatmap, pred_y):
        plt.figure(figsize=(8, 8))
        plt.imshow(logits_heatmap, cmap="viridis", origin="lower", alpha=1, interpolation="bicubic")
        plt.imshow(x, cmap="gray", origin="lower", alpha=0.3)
        plt.plot(y[0], y[1], label="true")
        plt.plot(pred_y[0], pred_y[1], label="predicted")
        plt.xlim([0, IMSIZE - 1])
        plt.ylim([0, IMSIZE - 1])
        # bbox_to_anchor defines how far the legend is from the anchor,
        # even though its upper left, 1.02 moves it a bit out of the plot
        plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", edgecolor="black", fancybox=False)
        plt.colorbar(label="Logit Probability")
        plt.show()

    def ask(self):
        self.params = np.array(self.es.ask(), dtype=np.float32)
        return self.params

    def tell(self, loss):
        loss = np.array(loss, dtype=np.float32)
        self.es.tell(self.params, loss.tolist())

    def test(self, champion_idx):
        train_x, train_y = BezierSolver.gen_train_data()
        num_iters = 10
        for i in range(min(num_iters, len(train_x))):
            x = np.reshape(train_x[i], (IMSIZE, IMSIZE))
            y = np.transpose(np.reshape(train_y[i], (NUM_POINTS, P_DIM))) * IMSIZE
            logits = BezierSolver.mlp_forward(self.params[champion_idx], train_x[i])
            logits = scipy.special.softmax(logits)
            coords = BezierSolver.logits_to_coords(logits)
            pred_y = np.transpose(np.reshape(coords, (NUM_POINTS, P_DIM))) * IMSIZE
            logits_heatmap = np.reshape(logits, (IMSIZE, IMSIZE)).transpose()
            BezierSolver.plot_heatmap(x, y, logits_heatmap, pred_y)

    def train(self):
        num_gens = TOTAL_GENS
        print("Training...")
        for gen in range(num_gens):
            train_x, train_y = self.gen_train_data()
            np.ndarray(self.train_x_shape, dtype=train_x.dtype, buffer=self.train_x_shm.buf)[:] = train_x
            np.ndarray(self.train_y_shape, dtype=train_y.dtype, buffer=self.train_y_shm.buf)[:] = train_y
            print(f'Current memory usage: {memory_usage()}')
            self.ask()
            chunk_size = POP_SIZE // NUM_CORES
            with Pool(NUM_CORES) as pool:
                print(f'Current memory usage: {memory_usage()}')
                args = [(self.params[i:i + chunk_size],
                         self.train_x_shm.name,
                         self.train_y_shm.name,
                         self.train_x_shape,
                         self.train_y_shape) for i in range(0, POP_SIZE, chunk_size)]
                worker_loss = np.array(pool.starmap(BezierSolver.eval_params, args), dtype=np.float32)
                loss = worker_loss.ravel()
                print(f'Current memory usage: {memory_usage()}')
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
    # prev_params = np.load("./params_50.npy")
    # print(prev_params.shape)
    solver = BezierSolver()
    # solver.test(0)
    solver.train()
