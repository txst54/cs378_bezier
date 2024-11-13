import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Sequence, Tuple
import random
import time
import h5py
from utils import bezier
import tqdm

"""HYPERPARAMETERS"""
NUM_POINTS = 3  # n control points
P_DIM = 2  # n-dimensional bezier curve
IMSIZE = 28  # nxn image

debug = []
train_x = []
train_y = []


def gen_train_data():
    for i in tqdm.tqdm(range(100)):
        num_points = NUM_POINTS  # Number of points, Dimension of points
        p_dim = P_DIM
        x = [[random.random(), random.random()] for j in range(num_points)]
        num_points = num_points - 1
        bezier = bezier.eval_bezier(np.linspace(0, 1, IMSIZE), x)
        canvas = np.zeros((IMSIZE, IMSIZE))
        for j in range(len(bezier)):
            y = min(int(bezier[j][1] * IMSIZE), IMSIZE - 1)
            x = min(int(bezier[j][0] * IMSIZE), IMSIZE - 1)
            canvas[y][x] = 1
        debug.append(bezier)
        train_x.append(canvas)
        train_y.append(x)


