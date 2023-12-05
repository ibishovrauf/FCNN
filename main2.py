import numpy as np
from ConvLayer import Convolution2DLayer

s = Convolution2DLayer(3, 32, 1, 0, 1)
s.forward(np.random.randn(34, 34))