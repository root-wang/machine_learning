import numpy as np


def cost_function(x: np.array([], float), y: np.array, theta: list[float]):
    alpha = 0.001
    i = 0
    [theta0, theta1] = theta
    while i < 1000000:
        theta0 = theta0 - alpha * theta_loss(x, y, theta)
        theta1 = theta1 - alpha * theta_loss(x, y, theta)
        if np.abs(alpha * theta_loss(x, y, theta)) < 0.000001:
            print(i)
            break
        theta = [theta0, theta1]
        i = i + 1
    return theta


def theta_loss(x, y, theta):
    [theta0, theta1] = theta
    row, col = x.shape
    i = 0
    tmp = 0
    while i < row:
        tmp = tmp + (theta0 + theta1 * x[i, 0] - y[i, 0]) * x[i, 0]
        i = i + 1
    return tmp
