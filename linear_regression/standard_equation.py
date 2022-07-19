import numpy as np

m = 100

# 100*1均匀分布列向量
X = 2 * np.random.rand(m, 1)
# 100*1 标准正态分布列向量
Y = 345 + 33 * X  # added gaussian noise

x_b = np.c_[np.ones((m, 1)), X]
x_b_t = x_b.T
x_b_t_x_b_i = np.linalg.inv(np.dot(x_b_t, x_b))
theta = x_b_t_x_b_i.dot(x_b_t).dot(Y)

print(theta)
