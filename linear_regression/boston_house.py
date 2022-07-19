import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import linear_regression.utils.cost_function_vectory as cf

sns.set()

data_path = './data/housing-Sheet1.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'GE', 'DIS', 'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'PRICE']
boston_data = pd.read_csv(data_path, names=names, delim_whitespace=True)

boston_data.insert(0, 'Ones', 1)
col, row = boston_data.shape

X = np.mat(boston_data.iloc[:, 0:col - 1])
Y = np.mat(boston_data['PRICE']).T
theta = np.mat(np.zeros(row))

theta = cf.cost_function(X, Y, theta)
# 未正则化参数
# theta = np.mat(
#     [8.12141408e-03, 7.84875595e-04, 1.87587557e-03, 1.87957533e-03, 5.67176133e-03, 3.49386935e-03, 8.51092019e-02,
#      5.08295321e-03, 1.53918235e-03, 7.73866368e-03, - 6.56483741e-04, 4.85172597e-02,
#      -3.79359298e-04, - 5.81019002e-02, 9.65776230e-01])

# 正则过后的参数 theta = np.mat( [1.60091316e-05, 1.28969364e-06, 3.97036034e-06, 3.87281694e-06, 1.12112247e-05,
# 6.89596694e-06, 1.67792372e-04, 9.93897992e-06, 2.72855375e-06, 1.52161233e-05, -1.29933011e-06, 9.48064628e-05,
# -6.36674307e-07, -1.13173136e-04, 1.90715456e-03])
X_b = np.arange(0, col, 1)

plt.figure(figsize=(20, 20), dpi=150)
Y_b = X.dot(theta.T)
plt.plot(X_b, Y, 'bo')
plt.plot(X_b, Y_b, 'rx')

plt.show()
