# 编写神经网络class
import numpy as np

from nwpu.BackPropagation.neuralNetwroksLayer import NeuralNetworksLayer
from nwpu.BackPropagation.neuralNetworkNeurons import NeuralNetworkNeurons

from nwpu.BackPropagation.utils.derivative_sigmoid import derivative_sigmoid_fun


class NeuralNetworks:
    LEARNING_RATE = 0.001

    def __init__(self):
        self.layer_nums = 0
        self.layers = []

    # 计算一个样本的预测值
    def calculate_output(self, x):
        update_value = x
        for i in range(self.layer_nums):
            update_value = self.layers[i].calculate(update_value)
            self.layers[i].set_neural_network_neurons_value(update_value)
        return self.layers[-1].get_value_mat()

    def add_layer(self, input_num, neuron_num, activation):
        neuron_list = []
        for i in range(neuron_num):
            neuron = NeuralNetworkNeurons(np.random.rand())
            neuron_list.append(neuron)

        layer = NeuralNetworksLayer(input_num, neuron_list, activation)
        self.layer_nums = self.layer_nums + 1
        self.layers.append(layer)

    def backpropagation(self, x, y, learning_rate=None):
        output = self.calculate_output(x)
        print(output)
        for i in reversed(range(self.layer_nums)):
            layer = self.layers[i]
            if layer is self.layers[-1]:
                layer.error = output - y
                layer.delta = layer.error
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(layer.get_parameter(), next_layer.delta)
                layer.delta = np.multiply(layer.error, derivative_sigmoid_fun(
                    layer.get_value_mat()))
        for i in range(self.layer_nums):
            layer = self.layers[i]
            layer_output = x if i is 0 else self.layers[i - 1].get_value_mat()
            parameter = layer.get_parameter()
            delta = layer.delta
            update_parameter = parameter - np.multiply(learning_rate, np.dot(delta, layer_output.T))
            layer.set_parameter(update_parameter)

    def train_network(self, x_train, y_train, max_epochs, learning_rate):
        for i in range(max_epochs):
            for j in range(len(x_train)):
                self.backpropagation(x_train[j], y_train[j], learning_rate)
