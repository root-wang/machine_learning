#  神经网络层
import numpy as np
from nwpu.BackPropagation.utils.sigmoid import sigmoid_fun


class NeuralNetworksLayer:
    def __init__(self, neurons_num, neural_network_neurons, parameters_mat):
        self.neurons_num = neurons_num  # 该层的神经元数
        self.neural_network_neurons = neural_network_neurons  # 该层的所有神经元
        self.parameters_mat = parameters_mat

    def set_neural_network_neurons(self, neural_network_neurons):
        self.neural_network_neurons = neural_network_neurons

    # 计算该层的输入 通过了激活函数
    def calculate(self):
        parameter_mat = np.mat(self.get_parameter())
        value_mat = np.mat(self.get_value_mat())
        result = sigmoid_fun(np.matmul(parameter_mat, value_mat))
        return result

    def get_parameter(self):
        return self.parameters_mat

    def set_parameter(self, parameter_mat):
        self.parameters_mat = parameter_mat

    def get_value_mat(self, need_bias=True):
        value_list = []

        #  一层加入偏差1
        if need_bias:
            value_list.append(1)

        for neuron in self.neural_network_neurons:
            value_list.append(neuron.get_value())

        values = np.array(value_list)
        values = np.mat(values).T
        return values

    def set_neural_network_neurons_value(self, input_value):
        for i in range(1, self.neurons_num + 1):
            self.neural_network_neurons[i].set_value(input_value[0, i])
