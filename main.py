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
import contextlib


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
POP_SIZE = 6144
PARAM_SHAPE = (POP_SIZE, PARAM_SIZE)
INIT_STDEV = 0.2
NUM_SAMPLES = 32
TOTAL_GENS = 300
NUM_CORES = 32


def create_shm(params):
    """Create shared memory"""
    try:
        shm = shared_memory.SharedMemory(create=True, size=params.nbytes)
        np.ndarray(params.shape, dtype=params.dtype, buffer=shm.buf)[:] = params
        return shm.name
    finally:
        if 'shm' in locals():
            shm.close()


class BezierSolver:
    def __init__(self, params=None):
        # printf(f'Current memory usage: {memory_usage()}')
        if params is None:
            params = np.zeros(PARAM_SHAPE, dtype=np.float32)
        self.params_shm_name = create_shm(params)
        # printf(f'Current memory usage: {memory_usage()}')
        self.es = cma.CMAEvolutionStrategy(params[0],
                                           INIT_STDEV,
                                           {'popsize': POP_SIZE})
        # printf(f'Current memory usage: {memory_usage()}')

    @staticmethod
    @contextlib.contextmanager
    def get_params(shm_name, shape, dtype):
        """Safely retrieve parameters from shared memory."""
        try:
            params_shm = shared_memory.SharedMemory(name=shm_name)
            params = np.ndarray(shape, dtype=dtype, buffer=params_shm.buf)

            if params is None or params.size == 0:
                raise ValueError("Params are empty or not properly initialized.")

            yield params
        except Exception as e:
            print(f"Error retrieving params: {e}")
            return None
        finally:
            if 'params_shm' in locals():
                params_shm.close()

    @staticmethod
    def set_params(params_shm_name, new_params):
        with BezierSolver.get_params(params_shm_name, PARAM_SHAPE, np.float32) as params:
            params[:] = new_params

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
    def eval_params(shm_name, shape, dtype, start_idx, end_idx):
        with BezierSolver.get_params(shm_name, shape=shape, dtype=dtype) as params:
            if params is None:
                raise ValueError("Failed to retrieve params from shared memory.")
            params_chunk = params[start_idx:end_idx]
            train_x, train_y = BezierSolver.gen_train_data()
            losses = []

            for params in params_chunk:
                loss = 0
                # printf(f'Current memory usage: {memory_usage()}')
                for x, y in zip(train_x, train_y):
                    logits = BezierSolver.mlp_forward(params, x)
                    coords = BezierSolver.logits_to_coords(logits)
                    error = BezierSolver.mse(coords, y)
                    loss += error
                losses.append((loss / NUM_SAMPLES) * IMSIZE)
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
        params = np.array(self.es.ask(), dtype=np.float32)
        BezierSolver.set_params(self.params_shm_name, params)
        return params

    def tell(self, loss, params):
        params = np.array(params)
        self.es.tell(params, loss.tolist())

    def test(self, champion_idx):
        train_x, train_y = BezierSolver.gen_train_data()
        num_iters = 10
        with BezierSolver.get_params(self.params_shm_name, shape=PARAM_SHAPE, dtype=np.float32) as params:
            for i in range(min(num_iters, len(train_x))):
                x = np.reshape(train_x[i], (IMSIZE, IMSIZE))
                y = np.transpose(np.reshape(train_y[i], (NUM_POINTS, P_DIM))) * IMSIZE
                logits = BezierSolver.mlp_forward(params[champion_idx], train_x[i])
                logits = scipy.special.softmax(logits)
                coords = BezierSolver.logits_to_coords(logits)
                pred_y = np.transpose(np.reshape(coords, (NUM_POINTS, P_DIM))) * IMSIZE
                logits_heatmap = np.reshape(logits, (IMSIZE, IMSIZE)).transpose()
                BezierSolver.plot_heatmap(x, y, logits_heatmap, pred_y)

    def train(self):
        try:
            num_gens = TOTAL_GENS
            print("Training...")
            for gen in range(num_gens):
                # printf(f'Current memory usage: {memory_usage()}')
                params = self.ask()
                chunk_size = POP_SIZE // NUM_CORES
                with Pool(NUM_CORES) as pool:
                    # printf(f'Current memory usage: {memory_usage()}')
                    args = [(self.params_shm_name, PARAM_SHAPE, np.float32, i, i + chunk_size)
                            for i in range(0, POP_SIZE, chunk_size)]
                    worker_loss = np.array(pool.starmap(BezierSolver.eval_params, args), dtype=np.float32)
                    loss = worker_loss.ravel()
                    # printf(f'Current memory usage: {memory_usage()}')
                self.tell(loss, params)
                print(f'gen={gen}, '
                      f'best_loss={np.min(loss)}, '
                      f'avg_loss={np.mean(loss)}, '
                      f'champion_idx={np.argmin(np.array(loss))}'
                      )
                if gen % 10 == 0:
                    np.save(f'./params_{gen}.npy', params)
                del loss
                gc.collect()
        except KeyboardInterrupt:
            params_shm = shared_memory.SharedMemory(name=self.params_shm_name)
            params_shm.close()
            params_shm.unlink()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # prev_params = np.load("./params_50.npy")
    # print(prev_params.shape)
    solver = BezierSolver()
    # solver.test(0)
    solver.train()
