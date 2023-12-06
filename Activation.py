import numpy as np

class Activation:
    def __init__(self, activation, activation_der) -> None:
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_grad):
        return output_grad*self.activation_der(self.input)

