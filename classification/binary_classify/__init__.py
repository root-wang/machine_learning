import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

import classification.utils.gradient as gradient
import classification.utils.predict as predict

# 文件相对路径
data_path = ".././data_set/ex2data1.txt"
# names = [] 每一列的title
data = pd.read_csv(data_path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
# 列title是Admitted 数值是1的行
positive = data[data['Admitted'].isin([1])]
# 列title是Admitted 数值是0的行
negative = data[data['Admitted'].isin([0])]

# 一个图像 尺寸12*8 返回一个元组 第一个是figure
fig, ax = plt.subplots(figsize=(12, 8))
# ax是axes对象
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='blue', marker='x', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='red', marker='o', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')

# 数据处理
data.insert(0, "Ones", 1)

rows, cols = data.shape
X = np.mat(data.iloc[:, 0: cols - 1])
Y = np.mat(data.iloc[:, cols - 1:cols])
theta = np.array([[-25], [0.2], [0.2]])

theta = gradient.gradient_function(X, Y, theta)
theta0 = theta[0, 0]
theta1 = theta[1, 0]
theta2 = theta[2, 0]
X1 = np.linspace(data.Exam1.min(), data.Exam1.max(), 100)
X2 = (0.5 - theta0 - theta1 * X1) / theta2
plt.plot(X1, X2, 'r')
plt.show()

predictions = predict.predict_function(theta, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct))
print(accuracy)
