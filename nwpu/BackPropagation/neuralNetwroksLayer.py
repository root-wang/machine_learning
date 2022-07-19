#  神经网络层
import neuralNetworkNeurons


class NeuralNetworksLayer:
    def __init__(self, neurons_num, layers, neural_network_neurons):
        self.neurons_num = neurons_num  # 该层的神经元数
        self.layers = layers  # 第几层
        self.neural_network_neurons = neural_network_neurons  # 该层的所有神经元

    def calculate(self, neural_networks_layer):
        pass
