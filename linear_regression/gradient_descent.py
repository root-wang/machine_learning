import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import linear_regression.utils.cost_function_vectory as cf

# 梯度下降法

m = 100

# # 100*1均匀分布列向量
# # X = 2 * np.random.rand(m, 1)
# X = np.array([[1, 2, 6], [1, 0, 2], [1, 3, 4], [1, 9, 10]])
# # 100*1 标准正态分布列向量
# # Y = 345 + 33 * X  # added gaussian noise
# Y = np.array([[9, 3, 8, 20]])

# plt.figure()
# plt.plot(X.T[0], Y.T[0], 'ro')

dataPath = "./data/ex1data1.txt"
data = pd.read_csv(dataPath, header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))

data.insert(0, 'Ones', 1)
col, row = data.shape
Y = np.mat(data["Profit"]).T
X = np.mat(data.iloc[:, 0:row - 1])
theta = np.array([0, 0])

X_b = np.linspace(data.Population.min(), data.Population.max(), 100)

theta = cf.cost_function(X, Y, theta)

plt.plot(X_b, theta[0, 0] + theta[0, 1] * X_b, 'ro')
plt.show()
