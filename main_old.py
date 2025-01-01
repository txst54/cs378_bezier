import logging
import multiprocessing
import sys
import warnings
from typing import SupportsIndex, Sequence

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
import cv2

from utils.bezier import eval_bezier
from utils.shared_memory import set_shm, get_shm, create_shm, clean_shm, init_worker, memory_usage

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
TOTAL_TOURNAMENTS = 1000
NUM_CORES = 16

"""PREDEFINED MACROS"""
PARAM_SHAPE = Sequence[SupportsIndex](POP_SIZE, PARAM_SIZE)
TRAIN_X_SHAPE = Sequence[SupportsIndex](NUM_SAMPLES, IMSIZE * IMSIZE)
TRAIN_Y_SHAPE = Sequence[SupportsIndex](NUM_SAMPLES, NUM_POINTS, P_DIM)


class BezierSolver:
    def __init__(self, params=None, mode='es', debug_mem=False):
        self.debug_mem = debug_mem
        self.profile_memory()
        if params is None:
            if mode == 'ga':
                params = np.random.normal(size=PARAM_SHAPE).astype(np.float32) * 0.5
            else:
                params = np.zeros(PARAM_SHAPE, dtype=np.float32)
        self.mode = mode
        self.params_shm_name = create_shm(params)
        print("Created Shared Memory")

        train_x, train_y = self.gen_train_data()
        self.train_x_shm_name = create_shm(train_x)
        self.train_y_shm_name = create_shm(train_y)
        self.profile_memory()
        if mode == 'es':
            self.es = cma.CMAEvolutionStrategy(params[0],
                                               INIT_STDEV,
                                               {'popsize': POP_SIZE})
        self.profile_memory()

    def profile_memory(self):
        if self.debug_mem:
            print(f'Current memory usage: {memory_usage()}')

    @staticmethod
    def gen_train_data():
        # debug = []
        train_x = np.zeros(TRAIN_X_SHAPE, dtype=np.float32)
        train_y = np.zeros(TRAIN_Y_SHAPE, dtype=np.float32)
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
            train_y[i] = np.array(control_points)
        return train_x, train_y

    @staticmethod
    def logits_to_coords(logits):
        # Flatten the array and find the indices of the 3 highest values
        flattened_indices = np.argpartition(logits, -3)[-3:]
        # Convert the flattened indices back to 2D coordinates
        coords = np.array(np.unravel_index(flattened_indices, (IMSIZE, IMSIZE))).T / IMSIZE
        return coords

    @staticmethod
    def mse(pred, y):
        p_idxes = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)]
        min_mse = float('inf')
        for p_idx in p_idxes:
            p_pred = pred[p_idx, :]
            min_mse = min(min_mse, ((p_pred - y) ** 2).mean())
        return min_mse

    @staticmethod
    def mse_and_perm(pred, y):
        p_idxes = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)]
        min_mse = float('inf')
        best_perm = None
        for p_idx in p_idxes:
            p_pred = pred[p_idx, :]
            new_mse = ((p_pred - y) ** 2).mean()
            if new_mse < min_mse:
                min_mse = new_mse
                best_perm = p_idx
        return min_mse, pred[best_perm, :]

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
    def eval_params(param_shm_name, train_x_shm_name, train_y_shm_name, start_idx, end_idx):
        with (get_shm(param_shm_name, shape=PARAM_SHAPE, dtype=np.float32) as params,
              get_shm(train_x_shm_name, shape=TRAIN_X_SHAPE, dtype=np.float32) as train_x,
              get_shm(train_y_shm_name, shape=TRAIN_Y_SHAPE, dtype=np.float32) as train_y):
            if params is None:
                raise ValueError("Failed to retrieve params from shared memory.")
            params_chunk = params[start_idx:end_idx]
            losses = []

            for param in params_chunk:
                loss = 0
                # printf(f'Current memory usage: {memory_usage()}')
                for x, y in zip(train_x, train_y):
                    logits = BezierSolver.mlp_forward(param, x)
                    coords = BezierSolver.logits_to_coords(logits)
                    error = BezierSolver.mse(coords, y)
                    loss += error
                losses.append((loss / NUM_SAMPLES) * IMSIZE * IMSIZE)
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
        # plt.colorbar(label="Logit Probability")
        plt.show()

    def ask_es(self):
        assert self.mode == 'es', "BezierSolver.mode needs to be 'es' to call ask_es()"
        params = np.array(self.es.ask(), dtype=np.float32)
        set_shm(self.params_shm_name, val=params,
                shape=PARAM_SHAPE, dtype=np.float32)
        return params

    def tell_es(self, loss, params):
        assert self.mode == 'es', "BezierSolver.mode needs to be 'es' to call tell_es()"
        params = np.array(params)
        self.es.tell(params, loss.tolist())

    def test(self, champion_idx, params):
        train_x, train_y = BezierSolver.gen_train_data()
        num_iters = 10
        for i in range(min(num_iters, len(train_x))):
            x = np.reshape(train_x[i], (IMSIZE, IMSIZE))
            y = np.transpose(train_y[i]) * IMSIZE
            logits = BezierSolver.mlp_forward(params[champion_idx], train_x[i])
            logits = scipy.special.softmax(logits)
            coords = BezierSolver.logits_to_coords(logits)
            mse, fixed_coords = self.mse_and_perm(coords, train_y[i])
            print(f'MSE: {mse * IMSIZE * IMSIZE}')
            pred_y = np.transpose(fixed_coords) * IMSIZE
            logits_heatmap = np.reshape(logits, (IMSIZE, IMSIZE)).transpose()
            BezierSolver.plot_heatmap(x, y, logits_heatmap, pred_y)

    def train(self):
        if self.mode == 'es':
            self.train_es()
        else:
            self.train_ga()

    def train_es(self):
        try:
            num_gens = TOTAL_GENS
            print("Training with CMA-ES...")
            for gen in range(num_gens):
                # printf(f'Current memory usage: {memory_usage()}')
                params = self.ask_es()
                chunk_size = POP_SIZE // NUM_CORES
                with Pool(NUM_CORES) as pool:
                    # printf(f'Current memory usage: {memory_usage()}')
                    args = [(self.params_shm_name, PARAM_SHAPE, np.float32, i, i + chunk_size)
                            for i in range(0, POP_SIZE, chunk_size)]
                    worker_loss = np.array(pool.starmap(BezierSolver.eval_params, args), dtype=np.float32)
                    loss = worker_loss.ravel()
                    # printf(f'Current memory usage: {memory_usage()}')
                self.tell_es(loss, params)
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

    def run_tournament(self, loss, win_streak):
        with (get_shm(self.params_shm_name, shape=PARAM_SHAPE, dtype=np.float32) as params):
            idxes = np.array(list(range(POP_SIZE)))
            np.random.shuffle(idxes)
            pairs = np.reshape(idxes, (POP_SIZE // 2, 2))
            for pair in pairs:
                idx_l, idx_w = pair
                noise = np.random.normal(size=PARAM_SIZE).astype(np.float32) * 0.1
                if loss[idx_l] == loss[idx_w]:
                    params[idx_l] = params[idx_w] + noise
                    continue
                if loss[idx_l] < loss[idx_w]:
                    idx_l, idx_w = idx_w, idx_l
                params[idx_l] = params[idx_w] + noise
                win_streak[idx_l] = win_streak[idx_w]
                win_streak[idx_w] += 1
        return win_streak

    def draw(self, champion_idx, params):
        image = cv2.imread("./longhorn_px.jpg")
        # Convert the image to grayscale first (if it's not already in grayscale)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the grayscale image
        # The threshold value (127) can be adjusted based on your image
        _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
        # Calculate padding (equal padding on all sides)
        height, width = binary_image.shape
        max_dim = max(height, width)
        top = (max_dim - height) // 2
        bottom = max_dim - height - top
        left = (max_dim - width) // 2
        right = max_dim - width - left

        # Add padding using cv2.copyMakeBorder
        binary_image = cv2.copyMakeBorder(binary_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # plt.imshow(binary_image)
        # plt.show()
        self.recurse(image, binary_image, 1, binary_image, champion_idx, params)
        plt.imshow(image)
        plt.show()

    def recurse(self, color_image, binary_image, scale, og_img, champion_idx, params):
        if scale == 4:
            return
        print(f'Color img shape {color_image.shape}')
        reference = np.copy(binary_image)
        new_width = IMSIZE * 2
        new_height = IMSIZE * 2
        print(f'Bin img shape {binary_image.shape}')
        resized_image = cv2.resize(reference, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        _, resized_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
        print(f'Resized img shape {resized_image.shape}')
        # plt.imshow(resized_image)
        # plt.show()
        REAL_SIZE = binary_image.shape[0] // 2
        quads = [resized_image[:IMSIZE, :IMSIZE], resized_image[:IMSIZE, IMSIZE:], resized_image[IMSIZE:, :IMSIZE],
                 resized_image[IMSIZE:, IMSIZE:]]
        views = [color_image[:REAL_SIZE, :REAL_SIZE], color_image[:REAL_SIZE, REAL_SIZE:],
                 color_image[REAL_SIZE:, :REAL_SIZE], color_image[REAL_SIZE:, REAL_SIZE:]]
        refs = [reference[:REAL_SIZE, :REAL_SIZE], reference[:REAL_SIZE, REAL_SIZE:],
                reference[REAL_SIZE:, :REAL_SIZE], reference[REAL_SIZE:, REAL_SIZE:]]
        for q, v, r in zip(quads, views, refs):
            print(f'Color img shape {color_image.shape}')
            print(f'Bin img shape {binary_image.shape}')
            print(f'Quad shape {q.shape}')
            print(f'View shape {v.shape}')
            if (scale == 3):
                self.process_quadrant(q, scale, champion_idx, params, v)
            self.recurse(v, r, scale + 1, og_img, champion_idx, params)
        # plt.imshow(og_img)
        # plt.show()

    def process_quadrant(self, quadrant, scale, champion_idx, params, view):
        # Flatten and normalize the quadrant to pass to the MLP
        x = np.ravel(quadrant.astype(np.float32)) / 255
        print(scale)
        print(x.shape)
        logits = BezierSolver.mlp_forward(params[champion_idx], x)
        logits = scipy.special.softmax(logits)
        coords = BezierSolver.logits_to_coords(logits)
        ### PLEASE FIX
        real_image_size = 144 // pow(2, scale)

        # Evaluate Bezier curve and update the view based on the coordinates
        tt = np.linspace(0, 1, real_image_size)
        spline = eval_bezier(tt, np.array(coords))
        logits_heatmap = np.reshape(logits, (IMSIZE, IMSIZE)).transpose()
        # print(logits_heatmap)
        rgb_map = np.array(logits_heatmap[..., np.newaxis] * np.array([20000, 0, 20000]), dtype=np.uint8)
        print(rgb_map)
        rgb_map = cv2.resize(rgb_map, (18, 18), interpolation=cv2.INTER_LINEAR)
        view += rgb_map
        # Update the view based on the spline
        for j in range(len(spline)):
            y = min(int(spline[j][1] * real_image_size), real_image_size - 1)
            x = min(int(spline[j][0] * real_image_size), real_image_size - 1)
            # Mark the point as black (binary)
            # print(rgb_map)
            # view[y, x] = (255, 0, 0)

        ### END FIX

        return view

    def train_ga(self):
        try:
            print("Training with GA...")
            win_streak = np.zeros(POP_SIZE)
            for tournament in range(1, TOTAL_TOURNAMENTS + 1):
                with (get_shm(self.train_x_shm_name, TRAIN_X_SHAPE, np.float32) as train_x,
                      get_shm(self.train_y_shm_name, TRAIN_Y_SHAPE, np.float32) as train_y):
                    tx, ty = self.gen_train_data()
                    train_x[:] = tx
                    train_y[:] = ty
                # printf(f'Current memory usage: {memory_usage()}')
                chunk_size = POP_SIZE // NUM_CORES
                with Pool(NUM_CORES, initializer=init_worker) as pool:
                    # printf(f'Current memory usage: {memory_usage()}')
                    args = [(self.params_shm_name, self.train_x_shm_name, self.train_y_shm_name, i, i + chunk_size)
                            for i in range(0, POP_SIZE, chunk_size)]
                    worker_loss = np.array(pool.starmap(BezierSolver.eval_params, args), dtype=np.float32)
                    loss = worker_loss.ravel()
                    # printf(f'Current memory usage: {memory_usage()}')
                win_streak = self.run_tournament(loss, win_streak)
                record_holder = np.argmax(win_streak)
                print(f'tournament={tournament}, '
                      f'best_loss={np.min(loss)}, '
                      f'avg_loss={np.mean(loss)}, '
                      f'champion_idx={np.argmin(np.array(loss))}, '
                      f'best_win_streak={win_streak[record_holder]}, '
                      f'record_holder_idx={record_holder}'
                      )
                if tournament % 10 == 0:
                    with get_shm(self.params_shm_name, shape=PARAM_SHAPE, dtype=np.float32) as params:
                        np.save(f'./params-ga_{tournament}.npy', params)
                del loss
                gc.collect()
        except KeyboardInterrupt:
            clean_shm(self.params_shm_name)
            clean_shm(self.train_x_shm_name)
            clean_shm(self.train_y_shm_name)
        finally:
            clean_shm(self.params_shm_name)
            clean_shm(self.train_x_shm_name)
            clean_shm(self.train_y_shm_name)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    warnings.filterwarnings("ignore", message=".*Falling back to cpu.*")
    prev_params = np.load("./params-ga_300.npy")
    # print(prev_params)
    print(prev_params.shape)
    solver = BezierSolver(params=prev_params, mode='ga')
    solver.test(3, prev_params)
    # solver.draw(3, prev_params)
    # solver.train()
