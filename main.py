"""HYPERPARAMETERS"""
import multiprocessing
import warnings

import numpy as np

from losses.coords import CoordLoss
from losses.heatmap import HeatmapLoss
from models.convolution import ConvolutionalNeuralNetwork
from models.feedforward import FeedForward
from trainers.simple_ga import SimpleGATrainer

NUM_POINTS = 3  # n control points
P_DIM = 2  # n-dimensional bezier curve
IMSIZE = 12  # nxn image

INPUT_DIM = (IMSIZE * IMSIZE)
OUTPUT_DIM = (IMSIZE * IMSIZE)
POP_SIZE = 512
INIT_STDEV = 0.2
NUM_SAMPLES = 100
TOTAL_GENS = 300
TOTAL_TOURNAMENTS = 1000
NUM_CORES = 16

"""PREDEFINED MACROS"""
TRAIN_X_SHAPE = (NUM_SAMPLES, IMSIZE * IMSIZE)
TRAIN_Y_SHAPE = (NUM_SAMPLES, NUM_POINTS, P_DIM)

"""CNN ARCHITECTURE"""
CNN_PARAMS = [
    {
        "filters": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    {
        "filters": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    {
        "filters": 16,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    },
    {
        "filters": 32,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    }
]

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    warnings.filterwarnings("ignore", message=".*Falling back to cpu.*")
    prev_params = np.load("./zoo/conv-sga/params-ga_40.npy")
    # print(prev_params)
    print(prev_params.shape)

    # model = FeedForward(INPUT_DIM, OUTPUT_DIM)
    # loss_fn = HeatmapLoss(mode="stochastic", num_points=NUM_POINTS, imsize=IMSIZE)
    model = ConvolutionalNeuralNetwork(IMSIZE, NUM_POINTS * P_DIM, CNN_PARAMS, pool_shape=(1, 1))
    loss_fn = CoordLoss(num_points=NUM_POINTS, p_dim=P_DIM)
    solver = SimpleGATrainer(model=model,
                             loss=loss_fn,
                             num_points=NUM_POINTS,
                             p_dim=P_DIM,
                             imsize=IMSIZE,
                             num_train_samples=NUM_SAMPLES,
                             pop_size=POP_SIZE,
                             total_tournaments=TOTAL_TOURNAMENTS,
                             num_cores=NUM_CORES,
                             params=prev_params,
                             starting_param_idx=1)
    solver.train()
    # test_input = np.zeros((12, 12))
    # test_params = np.random.normal(size=(model.get_num_params())).astype(np.float32) * 0.5
    # y = model.forward(test_params, test_input)
    # print(model.get_num_params())
    # print(y)
    # print(loss_fn.loss(y, np.zeros((3, 2))))
    # solver.test(3, prev_params)
