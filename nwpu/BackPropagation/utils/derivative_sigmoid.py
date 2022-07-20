import numpy as np

from nwpu.BackPropagation.utils.sigmoid import sigmoid_fun


def derivative_sigmoid_fun(z):
    z = sigmoid_fun(z)
    return np.multiply(z, 1 - z)
