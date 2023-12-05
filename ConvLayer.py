import numpy as np


class Convolution2DLayer:
    def __init__(self, input_shape, kernel_size, depth, padding, type, stride) -> None:
        input_depth, input_height, input_weight = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_weight - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_size)
        self.bias = np.random.randn(*self.output_shape)
        self.padding = padding
        self.type = type
        self.stride = stride

    def forward(self, input):
        self.input = input
        output = np.copy(self.bias)

        for i in self.depth:
            for j in self.input_depth:
                pass
        # for filter_index in range(self.filters.shape[2]):
        #     filt = []
        #     for row in range(0, data.shape[1]+1-self.input_shape, self.stride):
        #         out = []
        #         for col in range(0, data.shape[0]+1-self.input_shape, self.stride):
        #             out.append(np.sum(data[row:row+3, col:col+3]*self.filters[:, :, filter_index]))
        #         filt.append(out)
        #     output.append(filt)
        # print(np.array(output).shape)


class Convolution1DLayer:
    def __init__(self, shape, filters, padding, stride) -> None:
        self.filters = np.random.random((filters, shape[0], shape[1], shape[2]))
        self.padding = padding
        self.type = type
        self.stride = stride

    def forward(self, data):
        return np.sum(data*self.filters, axis=1)


if __name__ == "__main__":
    conv = Convolution1DLayer((3, 1, 1), 32, 0, 1)
    data = np.random.random((3, 3, 3))
    print(data)
    conv.forward(data)