import numpy as np
class Model:
    def __init__(self, layers, activation, activation_der) -> None:
        self.layers = layers
        self.activation = activation
        self.activation_der = activation_der

    def predict(self, data):
        x = data
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def fit(self, x_train, y_train, loss, loss_der, epochs = 100):
        for _ in range(epochs):
            y_pred = self.predict(x_train)
            output_grad = loss_der(y_train, y_pred)
            print(loss(y_train, y_pred))

            for layer in self.layers[::-1]:
                output_grad = layer.backward(output_grad)
