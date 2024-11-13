from scipy.special import comb
import numpy as np


def eval_bernstein(i, num_points, t):
    val = comb(num_points, i) * t ** i * (1. - t) ** (num_points - i)
    return val


def eval_bezier(t, control_points):
    """
     xx = eval_bezier(t, control_points)

     Evaluates a Bézier curve for the points in control_points.

     Inputs:
      control_points is a list (or array) or 2D coords
      t is a number (or list of numbers) in [0,1] where you want to
        evaluate the Bézier curve

     Output:
      xx is the set of 2D points along the Bézier curve
    """
    control_points = np.array(control_points)
    num_points, p_dim = np.shape(control_points)  # Number of points, Dimension of points
    num_points = num_points - 1
    eval_points = np.zeros((len(t), p_dim))

    for i in range(num_points + 1):
        eval_points += np.outer(eval_bernstein(i, num_points, t), control_points[i])

    return eval_points
