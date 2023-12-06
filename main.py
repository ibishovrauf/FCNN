import numpy as np

from Activation import Activation
from Layer import Layer2
from Model import Model
from Optimazer import Adam


def relu(x):
    return np.maximum(-0.01 * x, x)


def relu_der(x):
    return np.where(x > 0, 1, -0.01)


def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


if __name__ == '__main__':
    model = Model([
        Layer2(2, 6, 2, Adam(0.01)),
        Activation(sigmoid, sigmoid_der),
        Layer2(6, 1, 2, Adam(0.01)),
    ])

    x_train = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0]
    ])
    y_train = np.array([[0], [0], [1], [1]])

    print(model.predict(x_train))

    for _ in range(1000):
        num_data_points = x_train.shape[0]
        indices = np.random.choice(num_data_points, size=3, replace=False)
        model.fit(
            x_train=x_train[indices],
            y_train=y_train[indices],
            loss=mse,
            loss_der=mse_prime,
            epochs=1
        )

    print(model.predict(x_train))
