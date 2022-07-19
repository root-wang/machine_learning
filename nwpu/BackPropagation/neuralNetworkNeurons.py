# 神经网络神经元 每个网络层的节点
import numpy as np


class NeuralNetworkNeurons:

    def __init__(self, value):
        self.value = value

    def set_value(self, value): self.value = value

    def get_value(self): return self.value


class NeuralNetworkNeuronsHidden(NeuralNetworkNeurons):

    def __init__(self, value=0, backend_weights=None, *kargs, **kwargs):
        super(NeuralNetworkNeuronsHidden, self).__init__(value)
        self.backend_weights = backend_weights
        # print(self.value)
        # print(self.backend_weights)
