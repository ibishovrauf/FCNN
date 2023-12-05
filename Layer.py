import numpy as np

class Layer:
    def __init__(self, input_shape, units, batch_size) -> None:
        self.w = np.random.randn(input_shape, units) * np.sqrt(2 / (input_shape + units))
        self.b = np.zeros((1, units))
        self.units = units
        self.batch_size = batch_size

    def forward(self, input):
        """
            input: shape(batch_size x number_of_features)
        """
        self.input = input
        self.output = np.dot(input, self.w)+self.b
        return self.output

    def backward(self, output_grad, learning_rate = 0.08):
        weight_grad = np.dot(np.array([np.mean(self.input, 0)]).T, output_grad) 
        input_grad = np.dot(output_grad, self.w.T)

        self.w -= learning_rate*weight_grad
        self.b -= learning_rate*np.sum(output_grad, axis=0, keepdims=True)
        return input_grad
    
# use any Optimazer
class Layer2:
    def __init__(self, input_shape, units, batch_size, optimazer) -> None:
        self.w = np.random.randn(input_shape, units) * np.sqrt(2 / (input_shape + units))
        self.b = np.zeros((1, units))
        self.m_w = np.zeros((input_shape, units))
        self.s_w = np.zeros((input_shape, units))
        self.m_b = np.zeros((units))
        self.s_b = np.zeros((units))
        self.units = units
        self.optimazer = optimazer
        self.batch_size = batch_size

    def forward(self, input):
        """
            input: shape(batch_size x number_of_features)
        """
        self.input = input
        self.output = np.dot(input, self.w)+self.b
        return self.output

    def backward(self, output_grad, learning_rate = 0.08):
        weight_grad = np.dot(np.array([np.mean(self.input, 0)]).T, output_grad) 
        input_grad = np.dot(output_grad, self.w.T)

        self.w, self.m_w, self.s_w = self.optimazer.update(weight_grad, self.w, self.m_w, self.s_w)
        self.w -= learning_rate*weight_grad
        self.b -= learning_rate*np.sum(output_grad, axis=0, keepdims=True)
        return input_grad
    
