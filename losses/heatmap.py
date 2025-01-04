import numpy as np

from losses.base_loss import BaseLoss


class HeatmapLoss(BaseLoss):
    def __init__(self, mode, num_points, imsize):
        """
        Arg 'mode' must be either 'stochastic' or 'deterministic
        """
        self.mode = mode
        self.imsize = imsize
        super().__init__(num_points)

    def logits_to_coords_deterministic(self, logits):
        """
        Converts probability heatmap logits to coordinates.
        Takes the 3 highest probabilities and extracts coordinates, returning a [num_points, p_dim] size array
        normalized to between [0, 1)
        """
        # Flatten the array and find the indices of the 3 highest values
        flattened_indices = np.argpartition(logits, -3)[-3:]
        # Convert the flattened indices back to 2D coordinates
        coords = np.array(np.unravel_index(flattened_indices, (self.imsize, self.imsize))).T / self.imsize
        return coords

    def logits_to_coords_stochastic(self, logits, num_samples=3):
        """
        Converts probability heatmap logits to coordinates.
        Samples 3 random probabilities and extracts coordinates, returning a [num_points, p_dim] size array
        normalized to between [0, 1)
        """
        # calculate softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        flat_probs = probabilities.ravel()
        flattened_indices = np.random.choice(len(flat_probs), size=num_samples, p=flat_probs, replace=False)
        # Convert the flattened indices back to 2D coordinates
        coords = np.array(np.unravel_index(flattened_indices, (self.imsize, self.imsize))).T / self.imsize
        return coords

    def loss(self, logits, y):
        if self.mode == "stochastic":
            coords = self.logits_to_coords_stochastic(logits)
        else:
            coords = self.logits_to_coords_deterministic(logits)
        return self.coords_mse(coords, y)
