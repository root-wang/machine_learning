import numpy as np

# node1 = neuralNetworkNeurons.NeuralNetworkNeuronsHidden(0.15, np.array([0.15, 0.25, 0.35]))
# node2 = neuralNetworkNeurons.NeuralNetworkNeuronsHidden(0.2, np.array([0.2, 0.3, 0.35]))
#
# layer = neuralNetwroksLayer.NeuralNetworksLayer(2, 2, [node1, node2])
from nwpu.BackPropagation import neurealNetwork

# from nwpu.BackPropagation.neuralNetwroksLayer import NeuralNetworksLayer
# from nwpu.BackPropagation.neuralNetworkNeurons import NeuralNetworkNeurons

if __name__ == "__main__":
    # 四个样本 每个样本两个特征
    inputs = np.mat([[0.05, 0.1], [0.15, 0.9], [0.3, 0.6], [0.85, 0.15]])
    # 四个输出分类
    outputs = np.array([[0.01, 0.99], [0.45, 0.23], [0.34, 0.56], [1.02, 0.45]])
    weight1 = np.array([[0.15, 0.25], [0.2, 0.3], [0.35, 0.35]])
    weight2 = np.array([[0.4, 0.5], [0.45, 0.55], [0.6, 0.6]])
    network = neurealNetwork.NeuralNetworks(inputs, outputs, [weight1, weight2])

    network.train_network()
    # print(network.calculate_output_error())

    # network.print()
