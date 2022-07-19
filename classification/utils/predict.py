import classification.utils.sigmoid as sigmoid
import numpy as np


def predict_function(theta, x):
    x = np.mat(x)
    theta = np.mat(theta)
    probability = sigmoid.sigmoid_fun(x.dot(theta))
    return [1 if x > 0.5 else 0 for x in probability]
