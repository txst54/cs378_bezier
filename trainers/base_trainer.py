import random
import time
from itertools import permutations
from multiprocessing import Pool

import numpy as np

from losses.base_loss import BaseLoss
from models.base_model import BaseModel
from utils import bezier
from utils.shared_memory import memory_usage, SharedMemoryManager, init_worker
from abc import ABC, abstractmethod


class TrainerWorker:
    """
    A lightweight worker that can be mapped across multiple cores concurrently for training
    (evaluating parameters)
    """

    def __init__(self, model: BaseModel, loss: BaseLoss, imsize: int, num_points: int, p_dim: int,
                 num_train_samples: int, param_shape: int, shm_manager: SharedMemoryManager):
        self.model = model
        self.loss_fn = loss
        self.imsize = imsize
        self.num_points = num_points
        self.p_dim = p_dim
        self.num_train_samples = num_train_samples
        self.train_x_shape = (num_train_samples, imsize * imsize)
        self.train_y_shape = (num_train_samples, num_points, p_dim)
        self.param_shape = param_shape
        self.shm_manager = shm_manager

    def eval_params(self, param_shm_name, train_x_shm_name, train_y_shm_name, start_idx, end_idx):
        with (self.shm_manager.get_shm(param_shm_name, shape=self.param_shape, dtype=np.float32) as params,
              self.shm_manager.get_shm(train_x_shm_name, shape=self.train_x_shape, dtype=np.float32) as train_x,
              self.shm_manager.get_shm(train_y_shm_name, shape=self.train_y_shape, dtype=np.float32) as train_y):
            if params is None:
                raise ValueError("Failed to retrieve params from shared memory.")
            params_chunk = params[start_idx:end_idx]
            losses = []

            for param in params_chunk:
                loss = 0
                # replace feedforward params with intializer values
                mlp_param = param[:self.model.conv_size]
                hebbian_param = param[self.model.conv_size:]
                mlp_param = np.concatenate((mlp_param, np.random.randn(self.model.mlp_size) * 0.01), axis=0)
                # printf(f'Current memory usage: {memory_usage()}')
                for x, y in zip(train_x, train_y):
                    logits, mlp_param = self.model.forward(mlp_param, x, hebbian_param)
                    error = self.loss_fn.loss(logits, y)
                    loss += error
                losses.append((loss / self.num_train_samples) * self.imsize * self.imsize)
            return losses


class BaseTrainer(ABC):

    def __init__(self,
                 model: BaseModel,
                 loss: BaseLoss,
                 num_points: int,
                 p_dim: int,
                 imsize: int,
                 num_train_samples: int,
                 pop_size: int,
                 num_cores: int):
        """
        Initializes a Trainer class, wraps model and creates aux functions that make it easier to
        train evolutionary algorithms
        :param num_points: number of bezier control points to account for (i.e. 3)
        :param p_dim: dimension of bezier generation (i.e. 2 for 2-dim)
        :param imsize: width = height of image (i.e. 12)
        :param pop_size: population size
        :param num_train_samples: number of training samples
        """
        self.model = model
        self.loss_fn = loss
        self.num_points = num_points
        self.p_dim = p_dim
        self.imsize = imsize

        self.input_dim = imsize * imsize
        self.output_dim = imsize * imsize
        self.param_size = self.model.get_num_params()
        self.pop_size = pop_size
        self.param_shape = (pop_size, self.param_size)

        self.train_x_shape = (num_train_samples, imsize * imsize)
        self.train_y_shape = (num_train_samples, num_points, p_dim)
        self.num_train_samples = num_train_samples

        self.num_cores = num_cores

    def __init_shm__(self,
                     params=None,
                     debug_mem=False):
        """
        Subclasses must run this after __init__(), initializes all shared memory instances of Trainer
        """
        self.debug_mem = debug_mem
        self.profile_memory()
        self.shm_manager = SharedMemoryManager()
        self.params_shm_name = self.shm_manager.create_shm(params)
        print("[Trainer] Created Shared Memory")
        train_x, train_y = self.gen_train_data()
        self.train_x_shm_name = self.shm_manager.create_shm(train_x)
        self.train_y_shm_name = self.shm_manager.create_shm(train_y)
        self.worker = TrainerWorker(self.model, self.loss_fn, self.imsize, self.num_points,
                                    self.p_dim, self.num_train_samples, self.param_shape,
                                    self.shm_manager)
        with (self.shm_manager.get_shm(self.train_x_shm_name, self.train_x_shape, np.float32) as train_x,
              self.shm_manager.get_shm(self.train_y_shm_name, self.train_y_shape, np.float32) as train_y):
            print(f"[Trainer] Can access shm in superclass")
        self.profile_memory()

    def profile_memory(self):
        if self.debug_mem:
            print(f'Current memory usage: {memory_usage()}')

    def gen_train_data(self):
        # debug = []
        train_x = np.zeros(self.train_x_shape, dtype=np.float32)
        train_y = np.zeros(self.train_y_shape, dtype=np.float32)
        for i in range(self.num_train_samples):
            # Number of points, Dimension of points
            control_points = [[random.random() for __ in range(self.p_dim)] for ___ in range(self.num_points)]
            spline = bezier.eval_bezier(np.linspace(0, 1, self.imsize), control_points)
            canvas = np.zeros((self.imsize, self.imsize))
            for j in range(len(spline)):
                y = min(int(spline[j][1] * self.imsize), self.imsize - 1)
                x = min(int(spline[j][0] * self.imsize), self.imsize - 1)
                canvas[y][x] = 1
            # debug.append(bezier)
            train_x[i] = np.ravel(np.array(canvas))
            train_y[i] = np.array(control_points)
        return train_x, train_y

    def env_step_parallel(self):
        """
        Parallelizes the environment worker over multiple cores
        """
        chunk_size = self.pop_size // self.num_cores
        with Pool(self.num_cores, initializer=init_worker) as pool:
            self.profile_memory()
            args = [(self.params_shm_name, self.train_x_shm_name, self.train_y_shm_name, i, i + chunk_size)
                    for i in range(0, self.pop_size, chunk_size)]
            worker_loss = np.array(pool.starmap(self.worker.eval_params, args), dtype=np.float32)
            loss = worker_loss.ravel()
            self.profile_memory()
        return loss

    @abstractmethod
    def train(self):
        pass
