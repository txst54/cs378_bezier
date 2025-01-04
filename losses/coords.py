from losses.base_loss import BaseLoss


class CoordLoss(BaseLoss):
    def __init__(self, num_points, p_dim):
        self.p_dim = p_dim
        super().__init__(num_points=num_points)

    def loss(self, logits, y):
        logits = logits.reshape((self.num_points, self.p_dim))
        return self.coords_mse(logits, y)
