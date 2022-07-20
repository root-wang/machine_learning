#  神经网络层
import numpy as np
from nwpu.BackPropagation.utils.sigmoid import sigmoid_fun

 
class NeuralNetworksLayer:
    def __init__(self, neurons_num, layer_num, neural_network_neurons):
        self.neurons_num = neurons_num  # 该层的神经元数
        self.layers = layer_num  # 第几层
        self.neural_network_neurons = neural_network_neurons  # 该层的所有神经元

    # 计算该层的输入 通过了激活函数
    def calculate(self, front_layer):
        value = front_layer.get_value_mat()
        parameter = self.get_parameter_mat()
        net = np.matmul(value, parameter)

        return sigmoid_fun(net)

    def get_parameter_mat(self):
        parameters_list = []

        for neuron in self.neural_network_neurons:
            parameters_list.append(neuron.get_backend_weights())

        # 将每一层的每个神经元的后向参数矩阵合并为该层的参数矩阵
        parameters_mat = parameters_list[0]
        for i in range(1, len(parameters_list)):
            parameters_mat = np.hstack((parameters_mat, parameters_list[i]))

        return parameters_mat

    def get_value_mat(self, need_bias=True):
        value_list = []

        for neuron in self.neural_network_neurons:
            value_list.append(neuron.get_value())

        #  每一层加入偏差1
        if need_bias:
            value_list.append(1)

        value_mat = np.mat(np.array(value_list))
        return value_mat

    def set_neural_network_neurons(self, neural_network_neurons):
        self.neural_network_neurons = neural_network_neurons

    def set_neural_network_neurons_value(self, input_value):
        for i in range(self.neurons_num):
            self.neural_network_neurons[i].set_value(input_value[0, i])

    def update_parameter_mat(self, parameter_mat):
        pass
