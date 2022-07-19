# 编写神经网络class

class NeuralNetworks:
    LEARNING_RATE = 0.001

    def __init__(self, inputs, outputs, hidden_nums):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_nums = hidden_nums
