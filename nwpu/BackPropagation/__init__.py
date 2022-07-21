import numpy as np

from nwpu.BackPropagation import neurealNetwork

if __name__ == "__main__":
    # 四个样本 每个样本两个特征
    inputs = np.mat([[0.05], [0.1]])
    # 四个输出分类
    outputs = np.array([[0.01], [0.99]])
    weight1 = np.array([[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]])
    weight2 = np.array([[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]])
    network = neurealNetwork.NeuralNetworks(inputs, outputs, [weight1, weight2])

    network.train_network()
    # print(network.calculate_output_error())

    # network.print()
