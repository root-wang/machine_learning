import classification.utils.sigmoid as sigmoid
import numpy as np


# 计算梯度下降中的步长 损失函数的导数部分
def gradient_step(x, y, theta):
    x = np.mat(x)
    y = np.mat(y)
    theta = np.mat(theta)
    col, row = x.shape

    grad = np.zeros(row)

    error = sigmoid.sigmoid_fun(x.dot(theta)) - y

    for i in range(row):
        grad[i] = error.T.dot(x[:, i]) / col

    return grad


def gradient_function(x, y, theta):
    step = 1
    alpha = 0.01
    # adjust_alpha_arr = [10, 100, 1000, 10000, 100000, 1000000]
    while step < 10000000:
        step_value = np.mat(gradient_step(x, y, theta))
        # 固定步长学习率衰竭
        if step % 200 == 1:
            alpha = alpha * 0.8
        theta = theta - alpha * step_value.T
        # print('now this step is ', step)
        # print('now the step_value is ', np.sum(np.abs(step_value)))
        if (np.sum(np.abs(step_value))) < 5e-6:
            print('steps is ', step)
            break
        step = step + 1
    # print(theta)
    return theta
