import numpy as np

# node1 = neuralNetworkNeurons.NeuralNetworkNeuronsHidden(0.15, np.array([0.15, 0.25, 0.35]))
# node2 = neuralNetworkNeurons.NeuralNetworkNeuronsHidden(0.2, np.array([0.2, 0.3, 0.35]))
#
# layer = neuralNetwroksLayer.NeuralNetworksLayer(2, 2, [node1, node2])
from nwpu.BackPropagation import neurealNetwork

if __name__ == "__main__":
    inputs = [0.05, 0.1]
    outputs = [0.01, 0.99]
    weight1 = np.array([[0.15, 0.25], [0.2, 0.3], [0.35, 0.35]])
    weight2 = np.array([[0.4, 0.5], [0.45, 0.55], [0.6, 0.6]])
    network = neurealNetwork.NeuralNetworks(inputs, outputs, [weight1, weight2])

    network.update_layer_neuron()

    network.print()
