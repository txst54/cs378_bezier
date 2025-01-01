import jax.numpy as jnp
from models.base_model import BaseModel


class FeedForward(BaseModel):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, params, obs):
        x = obs
        ss = 0
        ee = self.input_dim * self.output_dim
        w_in = params[ss:ee].reshape(self.input_dim, self.output_dim)
        ss = ee
        ee = ss + self.output_dim
        bias = params[ss:ee]
        x = jnp.tanh(jnp.dot(w_in, x) + bias)
        return x
