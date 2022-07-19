import numpy as np
import classification.utils.sigmoid as sigmoid


def cost_function(x, y, theta):
    col = x.shape[0]
    x_m = np.mat(x)
    y_m = np.mat(y)
    theta_m = np.mat(theta)
    first = np.multiply(-y_m, np.log(sigmoid.sigmoid_fun(x_m.dot(theta_m))))
    second = np.multiply((1 - y_m), np.log(1 - sigmoid.sigmoid_fun(x_m.dot(theta_m))))
    return np.sum(first - second) / col
