from typing import List
import numpy as np


class MetaCartPoleControl(object):
    """A feedforward neural network, with Hebbian learning."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int]):
        """Initialization.

        Arguments:
          input_dim     - Input dimension. 4 in our CartPole example.
          hidden_dims   - Hidden dimension. E.g., [32, 32].
        """
        self.input_dim = input_dim
        self.output_dim = 1

        self.w_sizes = []
        self.num_params = 0
        self.hebbian_w_sizes = []
        self.num_hebbian_params = 0

        dim_in = input_dim
        for hidden_dim in hidden_dims:
            self.w_sizes.append((dim_in, hidden_dim))
            self.num_params += dim_in * hidden_dim
            dim_in = hidden_dim
        self.w_sizes.append((dim_in, self.output_dim))
        self.num_params += dim_in * self.output_dim

        # Initialize self.hebbian_w_sizes and self.num_hebbian_params
        # Your code here
        dim_in = input_dim
        for hidden_dim in hidden_dims:
            # excluding lr in hebbian_w_sizes
            self.hebbian_w_sizes.append((4, dim_in, hidden_dim))
            self.num_hebbian_params += dim_in * hidden_dim * 4 + 1
            dim_in = hidden_dim

        self.hebbian_w_sizes.append((4, dim_in, self.output_dim))
        self.num_hebbian_params += dim_in * self.output_dim * 4 + 1

        print(f'#params={self.num_params}')
        print(f'#hebbian_params={self.num_hebbian_params}')

        # MLP parameters are randomly sampled from Uniform(-1, 1)
        self.mlp_params = np.random.rand(self.num_params) * 2 - 1
        self.mlp_params_hist = []

    def seed(self, seed):
        np_random = np.random.RandomState(seed)
        self.mlp_params = np_random.rand(self.num_params) * 2 - 1
        self.mlp_params_hist = [self.mlp_params.copy()]

    def __call__(self,
                 hebbian_params: np.ndarray,
                 obs: np.ndarray):
        """Apply Hebbian learning rule to the mlp_params and also return action.

        Arguments:
          hebbian_params    - Hebbian learning parameters of shape (M,), where M
                              is the parameters' size.
          obs               - Network input data of shape (input_dim,).

        Returns:
          Action of shape (output_dim,).
        """
        assert hebbian_params.size == self.num_hebbian_params, (
            'Inconsistent params sizes.'
        )
        x = obs
        ss = 0
        h_ss = 0
        for w_size, hebbian_w_size in zip(self.w_sizes, self.hebbian_w_sizes):

            # Pass data through the MLP layer
            ee = ss + np.prod(w_size)
            w = self.mlp_params[ss:ee].reshape(w_size)
            pre_synaptic_activation = x
            x = np.tanh(np.einsum('i,ij->j', x, w))
            post_synaptic_activation = x

            # Apply the Hebbian learning rule
            # Your code here
            lr = hebbian_params[h_ss]
            h_ss += 1
            h_ee = h_ss + np.prod(hebbian_w_size)
            hebbian_w = hebbian_params[h_ss:h_ee]
            hebbian_w = hebbian_w.reshape(hebbian_w_size)
            for i in range(w_size[0]):
                for j in range(w_size[1]):
                    w[i][j] += lr * (hebbian_w[0][i][j] * pre_synaptic_activation[i] * post_synaptic_activation[j] +
                                     hebbian_w[1][i][j] * pre_synaptic_activation[i] +
                                     hebbian_w[2][i][j] * post_synaptic_activation[j] +
                                     hebbian_w[3][i][j])
            self.mlp_params[ss:ee] = w.flatten()

            ss = ee
            h_ss = h_ee

        assert ss == self.num_params
        assert h_ss == self.num_hebbian_params
        self.mlp_params_hist.append(self.mlp_params.copy())

        return 0 if x < 0 else 1