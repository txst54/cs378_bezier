import gc
import time
from multiprocessing import Pool
from typing import Tuple

import numpy as np

from losses.base_loss import BaseLoss
from models.base_model import BaseModel
from trainers.base_trainer import BaseTrainer
from utils.shared_memory import init_worker, SharedMemoryManager


class SimpleGATrainer(BaseTrainer):
    def __init__(self,
                 model: BaseModel,
                 loss: BaseLoss,
                 num_points: int,
                 p_dim: int,
                 imsize: int,
                 num_train_samples: int,
                 pop_size: int,
                 total_tournaments: int,
                 num_cores: int,
                 starting_param_idx=0,
                 params=None,
                 debug_mem=False
                 ):
        super().__init__(model, loss, num_points, p_dim, imsize,
                         num_train_samples, pop_size, num_cores)
        self.total_tournaments = total_tournaments
        if params is None:
            params = np.random.normal(size=self.param_shape).astype(np.float32) * 0.5
        else:
            noise = np.random.normal(size=(pop_size - 1, self.param_size)).astype(np.float32) * 0.1
            new_params = np.vstack([params[starting_param_idx], params[starting_param_idx] + noise])
            params = new_params
        super().__init_shm__(params, debug_mem)
        with (self.shm_manager.get_shm(self.train_x_shm_name, self.train_x_shape, np.float32) as train_x,
              self.shm_manager.get_shm(self.train_y_shm_name, self.train_y_shape, np.float32) as train_y,
              self.shm_manager.get_shm(self.params_shm_name, self.param_shape, np.float32) as params):
            print(f"[SimpleGA] Can access shm in subclass")

    def run_tournament(self, loss, win_streak):
        with (self.shm_manager.get_shm(self.params_shm_name, shape=self.param_shape, dtype=np.float32) as params):
            idxes = np.array(list(range(self.pop_size)))
            np.random.shuffle(idxes)
            pairs = np.reshape(idxes, (self.pop_size // 2, 2))
            for pair in pairs:
                idx_l, idx_w = pair
                noise = np.random.normal(size=self.param_size).astype(np.float32) * 0.01
                if loss[idx_l] == loss[idx_w]:
                    params[idx_l] = params[idx_w] + noise
                    continue
                if loss[idx_l] < loss[idx_w]:
                    idx_l, idx_w = idx_w, idx_l
                params[idx_l] = params[idx_w] + noise
                win_streak[idx_l] = win_streak[idx_w]
                win_streak[idx_w] += 1
        return win_streak

    def train(self):
        try:
            print("[SimpleGA] Training with GA...")
            win_streak = np.zeros(self.pop_size)
            for tournament in range(1, self.total_tournaments + 1):
                with (self.shm_manager.get_shm(self.train_x_shm_name, self.train_x_shape, np.float32) as train_x,
                      self.shm_manager.get_shm(self.train_y_shm_name, self.train_y_shape, np.float32) as train_y):
                    tx, ty = self.gen_train_data()
                    train_x[:] = tx
                    train_y[:] = ty
                self.profile_memory()
                start_time = time.time()
                loss = self.env_step_parallel()
                end_time = time.time()
                win_streak = self.run_tournament(loss, win_streak)
                record_holder = np.argmax(win_streak)
                print(f'tournament={tournament}, '
                      f'best_loss={np.min(loss):.2f}, '
                      f'avg_loss={np.mean(loss):.2f}, '
                      f'champion_idx={np.argmin(np.array(loss))}, '
                      f'best_win_streak={win_streak[record_holder]}, '
                      f'record_holder_idx={record_holder}, '
                      f'env_step_runtime={end_time - start_time:.2f} seconds'
                      )
                if tournament % 10 == 0:
                    with (self.shm_manager.get_shm(self.params_shm_name, shape=self.param_shape, dtype=np.float32)
                          as params):
                        np.save(f'./params-ga_{tournament}.npy', params)
                del loss
                gc.collect()
        except KeyboardInterrupt:
            self.shm_manager.clean_shm(self.params_shm_name)
            self.shm_manager.clean_shm(self.train_x_shm_name)
            self.shm_manager.clean_shm(self.train_y_shm_name)
        finally:
            self.shm_manager.clean_shm(self.params_shm_name)
            self.shm_manager.clean_shm(self.train_x_shm_name)
            self.shm_manager.clean_shm(self.train_y_shm_name)
