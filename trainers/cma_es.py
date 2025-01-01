import gc
from multiprocessing import Pool, shared_memory

import cma
import numpy as np

from trainers.base_trainer import BaseTrainer
from utils.shared_memory import set_shm, clean_shm


class CMAESTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 num_points: int,
                 p_dim: int,
                 imsize: int,
                 num_train_samples: int,
                 pop_size: int,
                 init_stdev: float,
                 total_gens: int,
                 num_cores: int,
                 params=None,
                 debug_mem=False):
        super().__init__(model, num_points, p_dim, imsize, num_train_samples, pop_size, num_cores)
        if params is None:
            params = np.zeros(self.param_shape, dtype=np.float32)
        super().__init_shm__(params, debug_mem)
        self.init_stdev = init_stdev
        self.total_gens = total_gens
        self.es = cma.CMAEvolutionStrategy(
            params[0],
            self.init_stdev,
            {'popsize': self.pop_size})
        self.profile_memory()

    def ask(self):
        params = np.array(self.es.ask(), dtype=np.float32)
        set_shm(self.params_shm_name, val=params,
                shape=self.param_shape, dtype=np.float32)
        return params

    def tell(self, loss, params):
        params = np.array(params)
        self.es.tell(params, loss.tolist())

    def train(self):
        try:
            num_gens = self.total_gens
            print("Training with CMA-ES...")
            for gen in range(num_gens):
                self.profile_memory()
                params = self.ask()
                loss = self.env_step_parallel()
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
            clean_shm(self.params_shm_name)
            clean_shm(self.train_x_shm_name)
            clean_shm(self.train_y_shm_name)
        finally:
            clean_shm(self.params_shm_name)
            clean_shm(self.train_x_shm_name)
            clean_shm(self.train_y_shm_name)
