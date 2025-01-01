import gc
from multiprocessing import Pool

import numpy as np

from trainers.base_trainer import BaseTrainer
from utils.shared_memory import get_shm, init_worker, clean_shm


class SimpleGATrainer(BaseTrainer):
    def __init__(self,
                 model,
                 num_points: int,
                 p_dim: int,
                 imsize: int,
                 num_train_samples: int,
                 pop_size: int,
                 total_tournaments: int,
                 num_cores: int,
                 params=None,
                 debug_mem=False
                 ):
        super().__init__(model, num_points, p_dim, imsize, num_train_samples, pop_size, num_cores)
        self.total_tournaments = total_tournaments
        if params is None:
            params = np.random.normal(size=self.param_shape).astype(np.float32) * 0.5
        super().__init_shm__(params, debug_mem)

    def run_tournament(self, loss, win_streak):
        with (get_shm(self.params_shm_name, shape=self.param_shape, dtype=np.float32) as params):
            idxes = np.array(list(range(self.pop_size)))
            np.random.shuffle(idxes)
            pairs = np.reshape(idxes, (self.pop_size // 2, 2))
            for pair in pairs:
                idx_l, idx_w = pair
                noise = np.random.normal(size=self.param_size).astype(np.float32) * 0.1
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
            print("Training with GA...")
            win_streak = np.zeros(self.pop_size)
            for tournament in range(1, self.total_tournaments + 1):
                with (get_shm(self.train_x_shm_name, self.train_x_shape, np.float32) as train_x,
                      get_shm(self.train_y_shm_name, self.train_y_shape, np.float32) as train_y):
                    tx, ty = self.gen_train_data()
                    train_x[:] = tx
                    train_y[:] = ty
                self.profile_memory()
                loss = self.env_step_parallel()
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
                    with get_shm(self.params_shm_name, shape=self.param_shape, dtype=np.float32) as params:
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
