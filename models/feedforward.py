import jax.numpy as jnp
from models.base_model import BaseModel


class FeedForward(BaseModel):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_params = self.input_dim * self.output_dim + self.output_dim

    def forward(self, params, obs, hebbian_params=None):
        x = obs
        ss = 0
        ee = self.input_dim * self.output_dim
        w_in = params[ss:ee].reshape(self.input_dim, self.output_dim)
        ss = ee
        ee = ss + self.output_dim
        bias = params[ss:ee]
        x = jnp.tanh(jnp.dot(w_in, x) + bias)
        return x

    def get_num_params(self):
        return self.n_params
