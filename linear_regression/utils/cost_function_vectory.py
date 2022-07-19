import numpy as np


def cost_function(x, y, theta):
    alpha = 0.00000001
    col, row = np.shape(x)
    i = 0
    lamda = 50
    while i < 10000000:
        theta = theta * (1 - alpha * lamda) - alpha * update_delta(x, y, theta).T
        print('第', i, '步')
        theta_error = np.sum(np.abs(alpha * update_delta(x, y, theta).T))
        print(theta_error)
        if theta_error < 0.00001:
            break
        i = i + 1
    return theta


def update_delta(x, y, theta):
    col, row = np.shape(x)
    tmp = []
    for sample in x:
        tmp.append(hypothesis_fun(sample, theta))
    tmp_m = tmp[0]
    for i in range(1, col):
        tmp_m = np.vstack([tmp_m, tmp[i]])
    m_1 = (tmp_m - np.mat(y)).T
    delta = {}
    for i in range(row):
        x_feature_sample = np.mat(x[:, i])  # col*1
        delta[i] = m_1.dot(x_feature_sample)
    delta_m = delta[0]
    for i in range(1, row):
        delta_m = np.vstack([delta_m, delta[i]])
    return delta_m


def hypothesis_fun(x, theta):
    theta = np.mat(theta)
    x = np.mat(x)
    return np.dot(theta, x.T)
