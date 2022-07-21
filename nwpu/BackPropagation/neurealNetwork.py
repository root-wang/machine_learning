# 编写神经网络class
import numpy as np

from nwpu.BackPropagation.neuralNetwroksLayer import NeuralNetworksLayer
from nwpu.BackPropagation.neuralNetworkNeurons import NeuralNetworkNeurons
import nwpu.BackPropagation.utils.derivative_sigmoid as ds


class NeuralNetworks:
    LEARNING_RATE = 0.001

    def __init__(self, inputs, outputs, weights_list):
        self.inputs = inputs
        self.outputs = outputs
        self.feature_nums = inputs.shape[1]
        self.output_nums = outputs.shape[1]
        self.weight_layer_nums: int = len(weights_list)  # 参数矩阵数 也是需要进行神经元计算的层数
        self.weights_list = weights_list
        self.predict = np.empty_like(self.outputs)
        self.layers = self.create_layer()

    # 忽略输入创建一个初始化的网络
    # 主要初始化参数矩阵 网络层数 神经元参数矩阵 隐藏层 输出层
    def create_layer(self):
        layers = []

        for i in range(len(self.weights_list)):
            weight = self.weights_list[i]
            # 根据权重矩阵得出该层和上一层的神经元数
            front_neural_nums, neural_nums = weight.shape
            neurons_list = []
            for k in range(front_neural_nums):
                neuron = NeuralNetworkNeurons(0)
                neurons_list.append(neuron)
            layer = NeuralNetworksLayer(neural_nums, neurons_list, weight)
            layers.append(layer)

            #     生成输出层
            neurons_list = []
            for k in range(self.output_nums):
                neuron = NeuralNetworkNeurons(0)
                neurons_list.append(neuron)
            layer = NeuralNetworksLayer(self.output_nums, neurons_list, None)
            layers.append(layer)
        return layers

    # 计算一个样本的预测值
    def calculate_layer_out(self):
        for i in range(0, self.weight_layer_nums):
            update_value = self.layers[i].calculate()
            self.layers[i + 1].set_neural_network_neurons_value(update_value)
        return self.layers[self.weight_layer_nums].get_value_mat(need_bias=False)

    # 计算输出层的误差
    def calculate_output_error(self):
        return self.predict - np.array(self.outputs)

    # 使用误差计算hidden layer 误差
    def calculate_hidden_error(self):
        pass

    def init_network_input(self, input_value):
        self.layers[0].set_neural_network_neurons_value(input_value)

    def train_network(self):
        # 输入样本数
        samples_num = self.inputs.shape[1]
        for i in range(samples_num):
            # 该样本的数据
            input_value = self.inputs[:, i]
            # 样本数据代入网络输入层 并进行计算
            self.init_network_input(input_value)
            self.predict[:, i] = self.calculate_layer_out()
            print(self.predict)
