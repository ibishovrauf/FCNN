from Layer import Layer2
from Model import Model
from Activation import Activation
from Optimazer import Adam
import numpy as np


def relu(x):
    return np.maximum(-0.01*x, x)

def relu_der(x):
    return np.where(x > 0, 1, -0.01)

def tanh(x):
    return np.tanh(x);

def tanh_der(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    sig = 1 / (1 + np.exp(-x))
    return sig*(1-sig)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

model = Model([
    Layer2(2, 6, 2, Adam(0.01)),
    Activation(sigmoid, sigmoid_der),
    Layer2(6, 1, 2, Adam(0.01)),
    Activation(sigmoid, sigmoid_der),
], sigmoid, sigmoid_der)

data = np.array([[1.2, 2.3, 6.5, 1.2, 8.3, 1.2], [1.2, 2.2, 6.6, 2.0, 8.3, 1.2]])

x_train = np.array([[-1,-1], [1,1], [-1, 1], [1, -1]])
y_train = np.array([[-1], [-1], [1], [1]])

data = np.array([[[-1, -1], [-1, -1]], [[1, 1], [-1, -1]], [[-1, 1], [1, -1]], [[1, -1], [1, -1]]])

print(model.predict(x_train))

for _ in range(1000):
    for i in range(4):
        num_data_points = data.shape[0]
        random_data_points = data[np.random.choice(num_data_points, size=2, replace=False)]
        model.fit(random_data_points[:, 0], random_data_points[:, 1, 0].reshape(-1, 1), mse, mse_prime, 1)

print(model.predict(x_train))
