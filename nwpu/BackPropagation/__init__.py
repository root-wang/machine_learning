import numpy as np

from nwpu.BackPropagation import neurealNetwork
from nwpu.BackPropagation.neuralNetwroksLayer import NeuralNetworksLayer
from nwpu.BackPropagation.typings.activation import Activation

if __name__ == "__main__":
    # 四个样本 每个样本两个特征
    inputs = np.mat([[0.05], [0.1]])
    # 四个输出分类
    outputs = np.array([[0.1], [0.199]])
    network = neurealNetwork.NeuralNetworks()
    network.add_layer(2, 4, Activation.SIGMOID.value)
    network.add_layer(4, 2, Activation.SIGMOID.value)

    network.train_network([inputs], [outputs], 10000, 0.05)
    # print(network.calculate_output_error())

    # network.print()
