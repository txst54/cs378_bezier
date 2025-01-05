from typing import List

import cma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from base64 import b64encode
import gym
from multiprocessing import Pool

import warnings

from hebbian.hebbian_cartpole_control import MetaCartPoleControl
from hebbian.hebbian_utils import nn_control, play_video

"""HYPERPARAMETERS"""
seed = 42
n_repeats = 10  # @param
num_gen = 10  # @param
pop_size = 32  # @param
init_stdev = 0.1  # @param
num_worker = 12  # @param
hidden_dims = [32, 32]  # @param
hebbian_model = MetaCartPoleControl(input_dim=4, hidden_dims=hidden_dims)
rand_hebbian_params = np.random.randn(hebbian_model.num_hebbian_params) * 0.01


def eval_hebbian_params(args):
    params, seed = args
    rewards = [nn_control(hebbian_model, params, seed + 37 * i, render=False)
               for i in range(n_repeats)]
    # We evaluate the parameters for multiple times to reduce noise.
    return np.mean(rewards)


def random_policy():
    # Control cart with a random Hebbian policy
    images = nn_control(hebbian_model, rand_hebbian_params, seed)
    print('Control the cart with a random Hebbian policy')
    play_video(images)


def train_hebbian():
    # Initialize the CMA-ES solver.
    algo = cma.CMAEvolutionStrategy(
        x0=np.zeros(hebbian_model.num_hebbian_params),
        sigma0=init_stdev,
        inopts={
            "popsize": pop_size,
            "seed": seed,
            "randn": np.random.randn,
        }
    )

    # Optimization loop, we use multiprocessing to accelerate the rollouts.
    with Pool(num_worker) as p:
        for i in range(num_gen * 5):
            print("asking")
            population = algo.ask()
            rollout_seed = np.random.randint(0, 10000000)
            print("starting evaluation of params")
            scores = p.map(eval_hebbian_params,
                           [x for x in zip(
                               [np.array(population[k]) for k in range(pop_size)],
                               [rollout_seed] * pop_size)])
            print("telling")
            algo.tell(population, [-x for x in scores])  # CMA-ES minimizes.
            # if i % 10 == 0:
            print(f'Gen={i + 1}, reward.max={np.max(scores)}')

    # Test and visualize the trained control policy.
    best_params = np.array(algo.result.xfavorite)
    np.save(f'./params-hebbian-best.npy', best_params)
    images = nn_control(hebbian_model, best_params, seed)
    play_video(images)


def visualize_hebbian():
    best_params = np.load("./params-hebbian-best.npy")
    images = nn_control(hebbian_model, best_params, seed, True, True)
    play_video(images)

    mlp_params_hist = np.array(hebbian_model.mlp_params_hist)
    print(mlp_params_hist.shape)
    plt.figure(figsize=(10, 5))
    plt.imshow(mlp_params_hist)
    _ = plt.colorbar()
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # random_policy()
    train_hebbian()
    # visualize_hebbian()
