import numpy as np


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-1 * x))
