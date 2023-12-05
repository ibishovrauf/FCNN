import numpy as np


class Adam:
    def __init__(self, lr, beta1= 0.9, beta2 = 0.99) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 10**(-8)

    def update(self, gradients, parameters, m, s):
        m = self.beta1 * m - (1-self.beta1) * gradients
        s = self.beta2 * s + (1-self.beta2) * np.square(gradients)
        m_hat = m/(1-self.beta1)
        s_hat = s/(1-self.beta1)
        parameters = parameters + self.lr*m_hat/(np.sqrt(s_hat)+self.epsilon)
        return parameters, m, s 