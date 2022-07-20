# 编写神经网络class
import neuralNetwroksLayer
import neuralNetworkNeurons


class NeuralNetworks:
    LEARNING_RATE = 0.001

    def __init__(self, inputs, outputs, weights_list):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layer_nums: int = len(weights_list)
        self.weights_list = weights_list
        self.layers = self.create_layer()

    def create_layer(self):
        layers = []

        # 创建输入层
        input_neurons_list = []
        for k in range(len(self.inputs)):
            input_neuron = neuralNetworkNeurons.NeuralNetworkNeurons(self.inputs[k])
            input_neurons_list.append(input_neuron)
        layer = neuralNetwroksLayer.NeuralNetworksLayer(len(self.inputs), 1, input_neurons_list)
        layers.append(layer)

        for i in range(len(self.weights_list)):
            weight = self.weights_list[i]
            # 根据权重矩阵得出该层和上一层的神经元数
            front_neural_nums, neural_nums = weight.shape
            neurons_list = []
            for k in range(neural_nums):
                neuron = neuralNetworkNeurons.NeuralNetworkNeuronsHidden(0, weight[:, k:k + 1])
                neurons_list.append(neuron)
            layer = neuralNetwroksLayer.NeuralNetworksLayer(neural_nums, i + 2, neurons_list)
            layers.append(layer)
        return layers

    def calculate_layer_out(self, layer_num):
        layer = self.layers[layer_num - 1]
        front_layer = self.layers[layer_num - 2]
        return layer.calculate(front_layer)

    def update_layer_neuron(self):
        for i in range(1, self.hidden_layer_nums + 1):
            # 神经网络的一层
            layer = self.layers[i]
            # 需要更新的节点值
            update_layer_neurons = self.calculate_layer_out(i + 1)
            # 为该层每一个节点(不包括偏置)更新数据
            for k in range(len(layer.neural_network_neurons)):
                neuron = layer.neural_network_neurons[k]
                neuron.set_value(update_layer_neurons[0, k])

    def print(self):
        for layer in self.layers:
            print('the neuron of the layer value is ')
            for neuron in layer.neural_network_neurons:
                print(neuron.get_value())
