import jax.numpy as jnp
import jax
from models.base_model import BaseModel


def conv2d(x, kernel, bias, stride=1):
    """
    Conv2D in JAX.

    Args:
        x (jax.numpy.DeviceArray): Input array of shape (batch_size, height, width, channels).
        kernel (np.ndarray): Kernel weights of shape (kernel_size, kernel_size, prev_channels, filter_size)
        bias (np.ndarray): Bias of shape (filter_size)
        stride (int): Size of stride

    Returns:
        jax.numpy.DeviceArray: Conv2D of input of shape (batch_size, target_height, target_width, filter_size).
    """
    print(x.shape)
    print(kernel.shape)
    return jax.lax.conv_general_dilated(
        x, kernel, window_strides=(stride, stride), padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
    ) + bias


def adaptive_avg_pool_2d(x, output_size):
    """
    Adaptive Average Pooling for 2D inputs in JAX.

    Args:
        x (jax.numpy.DeviceArray): Input array of shape (batch_size, height, width, channels).
        output_size (tuple): Desired output size (target_height, target_width).

    Returns:
        jax.numpy.DeviceArray: Pooled array of shape (batch_size, target_height, target_width, channels).
    """
    batch_size, input_height, input_width, channels = x.shape
    target_height, target_width = output_size

    # Calculate the pooling window size
    stride_height = input_height // target_height
    stride_width = input_width // target_width

    kernel_height = input_height - (target_height - 1) * stride_height
    kernel_width = input_width - (target_width - 1) * stride_width

    # Apply average pooling
    return jax.lax.reduce_window(
        x,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, kernel_height, kernel_width, 1),
        window_strides=(1, stride_height, stride_width, 1),
        padding="VALID",
    ) / (kernel_height * kernel_width)


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(self, input_dim, output_dim, param_shapes, pool_shape):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.param_shapes = param_shapes
        self.pool_shape = pool_shape
        self.n_params = self.__calc_param_size__()
        self.feed_forward_range = (0, 0)

    def __calc_param_size__(self):
        prev_channels = 1
        kernel_sizes = []
        bias_sizes = []
        for p_shape in self.param_shapes:
            k = p_shape["kernel_size"]
            kernel_sizes.append(k * k * prev_channels * p_shape["filters"])
            bias_sizes.append(p_shape["filters"])
            prev_channels = p_shape["filters"]
        w_size = prev_channels * self.output_dim
        b_size = self.output_dim
        self.conv_size = sum(kernel_sizes) + sum(bias_sizes)
        self.mlp_size = w_size + b_size
        self.feed_forward_range = (self.conv_size, w_size + b_size)
        self.hebbian_shape = (4, prev_channels, self.output_dim)
        return self.conv_size + w_size + b_size

    def __extract_weights__(self, params):
        kernels, biases = [], []
        ss = 0
        prev_channels = 1
        for p_shape in self.param_shapes:
            k = p_shape["kernel_size"]
            kernel_size = k * k * prev_channels * p_shape["filters"]
            ee = ss + kernel_size
            kernels.append(params[ss:ee].reshape(k, k, prev_channels, p_shape["filters"]))
            bias_size = p_shape["filters"]
            biases.append(params[ee:ee + bias_size])
            ss = ee + bias_size
            prev_channels = p_shape["filters"]
        ee = ss + prev_channels * self.output_dim
        w_in = params[ss:ee].reshape(prev_channels, self.output_dim)
        bias = params[ee:ee + self.output_dim]
        return kernels, biases, w_in, bias, ss, ee

    def forward(self, params, obs, hebbian_params=None):
        x = obs.reshape(1, self.input_dim, self.input_dim, 1)
        kernels, biases, w_in, b, ss, ee = self.__extract_weights__(params)
        for kernel, bias, p_shape in zip(kernels, biases, self.param_shapes):
            x = conv2d(x, kernel, bias, stride=p_shape["stride"])
            x = jax.nn.relu(x)
        x = adaptive_avg_pool_2d(x, self.pool_shape)
        x = x.squeeze()
        pre_synaptic_activation = x
        x = jax.nn.sigmoid(jnp.dot(w_in.T, x) + b)
        post_synaptic_activation = x
        if hebbian_params is not None:
            h_ss = 0
            lr = hebbian_params[h_ss]
            h_ss += 1
            hebbian_w_size = self.hebbian_shape
            h_ee = h_ss + jnp.prod(jnp.array(hebbian_w_size))
            hebbian_w = hebbian_params[h_ss:h_ee]
            hebbian_w = hebbian_w.reshape(hebbian_w_size)
            for i in range(w_in.shape[0]):
                for j in range(w_in.shape[1]):
                    w_in[i][j] += (lr * (hebbian_w[0][i][j] * pre_synaptic_activation[i] * post_synaptic_activation[j] +
                                 hebbian_w[1][i][j] * pre_synaptic_activation[i] +
                                 hebbian_w[2][i][j] * post_synaptic_activation[j] +
                                 hebbian_w[3][i][j]))
            params[ss:ee] = w_in.flatten()
        return x, params

    def get_num_params(self):
        # conv size + hebbian paramters + learning rate
        return self.conv_size + self.hebbian_shape[0] * self.hebbian_shape[1] * self.hebbian_shape[2] + 1
