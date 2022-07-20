# 编写神经网络class
import numpy as np

from nwpu.BackPropagation.neuralNetwroksLayer import NeuralNetworksLayer
from nwpu.BackPropagation.neuralNetworkNeurons import NeuralNetworkNeurons, NeuralNetworkNeuronsHidden
import nwpu.BackPropagation.utils.derivative_sigmoid as ds


class NeuralNetworks:
    LEARNING_RATE = 0.001

    def __init__(self, inputs, outputs, weights_list):
        self.inputs = inputs
        self.outputs = outputs
        self.feature_nums = inputs.shape[1]
        self.weight_layer_nums: int = len(weights_list)  # 参数矩阵数 也是需要进行神经元计算的层数
        self.weights_list = weights_list
        self.predict = np.empty_like(self.outputs)
        self.layers = self.create_layer()

    # 忽略输入创建一个初始化的网络
    # 主要初始化参数矩阵 网络层数 神经元参数矩阵 隐藏层 输出层
    def create_layer(self):
        layers = []

        # 创建输入层
        input_neurons_list = []
        for k in range(self.feature_nums):
            input_neuron = NeuralNetworkNeurons(0)
            input_neurons_list.append(input_neuron)
        layer = NeuralNetworksLayer(self.feature_nums, 1, input_neurons_list)
        layers.append(layer)

        for i in range(len(self.weights_list)):
            weight = self.weights_list[i]
            # 根据权重矩阵得出该层和上一层的神经元数
            front_neural_nums, neural_nums = weight.shape
            neurons_list = []
            for k in range(neural_nums):
                neuron = NeuralNetworkNeuronsHidden(0, weight[:, k:k + 1])
                neurons_list.append(neuron)
            layer = NeuralNetworksLayer(neural_nums, i + 2, neurons_list)
            layers.append(layer)
        return layers

    # 计算一个样本的预测值
    def calculate_layer_out(self):
        for i in range(self.weight_layer_nums):
            update_value = self.layers[i + 1].calculate(self.layers[i])
            self.layers[i + 1].set_neural_network_neurons_value(update_value)
        return self.layers[self.weight_layer_nums].get_value_mat(need_bias=False)

    # 计算输出层的误差
    def calculate_output_error(self):
        return self.predict - np.array(self.outputs)

    def print_layer_value(self):
        for layer in self.layers:
            print('the neuron of the layer value is ')
            for neuron in layer.neural_network_neurons:
                print(neuron.get_value())

    # 使用误差计算hidden layer 误差
    def calculate_hidden_error(self):
        # 隐藏层是参数矩阵数减一
        hidden_nums = self.weight_layer_nums - 1

        delta = [self.calculate_hidden_error()]
        # 从后往前计算
        for i in range(hidden_nums, 2, -1):
            # 获取隐藏层的原始参数矩阵
            weight_mat = self.layers[i].get_parameter_mat()
            values = self.layers[i].get_value_mat()
            error = np.multiply(np.matmul(delta[i - hidden_nums], weight_mat),
                                ds.derivative_sigmoid_fun(values))
            delta.append(error)
        return delta

    def update_weights(self, layer_num, error_mat):
        values = self.layers[layer_num].get_value_mat()
        error_mat = np.mat(error_mat)
        result = np.matmul(error_mat.T, values)
        return result

    def init_network_input(self, input_value):
        self.layers[0].set_neural_network_neurons_value(input_value)

    def train_network(self):
        # 输入样本数
        samples_num = self.inputs.shape[0]
        for i in range(samples_num):
            # 该样本的数据
            input_value = self.inputs[i, :]
            # 样本数据代入网络输入层 并进行计算
            self.init_network_input(input_value)
            self.predict[i, :] = self.calculate_layer_out()
        error = self.calculate_output_error()
        #         测试一个样本的训练
        error = error[0, :]
        for layer_num in range(self.weight_layer_nums, 0, -1):
            weight = self.layers[layer_num].get_parameter_mat()
            update_weights = self.update_weights(layer_num, error).T
            weight = weight - np.multiply(self.LEARNING_RATE, update_weights)

