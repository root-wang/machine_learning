#  神经网络层
import numpy as np
from nwpu.BackPropagation.utils.sigmoid import sigmoid_fun
from nwpu.BackPropagation.utils.derivative_sigmoid import derivative_sigmoid_fun
from nwpu.BackPropagation.typings.activation import Activation


class NeuralNetworksLayer:
    def __init__(self, input_num, neural_network_neurons, bias=None, activation=None):
        self.neural_network_neurons = neural_network_neurons  # 该层的所有神经元
        self.neuron_num = len(self.neural_network_neurons)
        self.parameters_mat = np.random.randn(self.neuron_num, input_num) * np.sqrt(1 / self.neuron_num)
        self.bias = bias if bias is None else np.random.randn(1) * 0.01
        self.activation = activation

    def set_neural_network_neurons(self, neural_network_neurons):
        self.neural_network_neurons = neural_network_neurons

    # 计算该层的输入 通过了激活函数
    def calculate(self, x):
        parameter_mat = np.mat(self.get_parameter())
        value_mat = np.mat(x)
        result = np.matmul(parameter_mat, value_mat)
        result = np.add(result, np.mat(self.bias))
        result = self.activation_fun(result)
        return result

    def get_parameter(self):
        return self.parameters_mat

    def set_parameter(self, parameter_mat):
        self.parameters_mat = parameter_mat

    def get_value_mat(self):
        value_list = []

        for neuron in self.neural_network_neurons:
            value_list.append(neuron.get_value())

        values = np.array(value_list)
        values = np.mat(values).T
        return values

    def set_neural_network_neurons_value(self, input_value):
        for i in range(len(self.neural_network_neurons)):
            value = input_value[i, 0]
            self.neural_network_neurons[i].set_value(value)

    def activation_fun(self, z):
        if self.activation is None:
            return z
        elif self.activation is Activation.RELU.value:
            return np.maximum(z, 0)
        elif self.activation is Activation.SIGMOID.value:
            return sigmoid_fun(z)

    def activation_derivative_fun(self, z):
        if self.activation is None:
            return np.ones_like(z)
        elif self.activation is Activation.RELU.value:
            grad = np.array(z)
            grad[z > 0] = 1
            grad[z <= 0] = 0
            return grad
        elif self.activation is Activation.SIGMOID.value:
            return derivative_sigmoid_fun(z)
