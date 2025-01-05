from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, input_dim, output_dim):
        pass

    @abstractmethod
    def forward(self, params, obs, hebbian_params=None):
        pass

    @abstractmethod
    def get_num_params(self):
        pass
