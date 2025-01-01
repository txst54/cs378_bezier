"""HYPERPARAMETERS"""
import multiprocessing
import warnings

import numpy as np

from models.feedforward import FeedForward
from trainers.simple_ga import SimpleGATrainer

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
PARAM_SHAPE = (POP_SIZE, PARAM_SIZE)
TRAIN_X_SHAPE = (NUM_SAMPLES, IMSIZE * IMSIZE)
TRAIN_Y_SHAPE = (NUM_SAMPLES, NUM_POINTS, P_DIM)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    warnings.filterwarnings("ignore", message=".*Falling back to cpu.*")
    prev_params = np.load("./params-ga_300.npy")
    # print(prev_params)
    print(prev_params.shape)

    model = FeedForward(INPUT_DIM, OUTPUT_DIM)
    solver = SimpleGATrainer(model=model,
                             num_points=NUM_POINTS,
                             p_dim=P_DIM,
                             imsize=IMSIZE,
                             num_train_samples=NUM_SAMPLES,
                             pop_size=POP_SIZE,
                             total_tournaments=TOTAL_TOURNAMENTS,
                             num_cores=NUM_CORES,
                             params=prev_params)
    solver.train()
    # solver.test(3, prev_params)
