from abc import ABC, abstractmethod
from itertools import permutations


class BaseLoss(ABC):
    def __init__(self, num_points):
        self.num_points = num_points

    @abstractmethod
    def loss(self, logits, y):
        pass

    def coords_mse(self, pred, y):
        p_idxes = list(permutations(list(range(self.num_points))))
        min_mse = float('inf')
        for p_idx in p_idxes:
            p_pred = pred[p_idx, :]
            min_mse = min(min_mse, ((p_pred - y) ** 2).mean())
        return min_mse
