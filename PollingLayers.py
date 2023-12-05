import numpy as np

class MaxPolling:
    def __init__(self, pool_size=(2, 2), strides=1) -> None:
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, data):
        self.data = data
        input_depth, input_height, input_width = data.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides, self.strides

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_array = np.zeros((input_depth, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                # Extract the region to pool
                window = data[:, i * stride_height: i * stride_height + pool_height,
                                    j * stride_width: j * stride_width + pool_width]
                
                # Apply max pooling along the depth dimension
                output_array[:, i, j] = np.amax(window, axis=(1, 2))

        return output_array
    

    def backward(self, output_grad):
        output = np.zeros(self.data.shape)
        input_depth, input_height, input_width = data.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides, self.strides

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_array = np.zeros(self.data.shape)

        for i in range(output_height):
            for j in range(output_width):
                window = data[:, i * stride_height: i * stride_height + pool_height,
                                    j * stride_width: j * stride_width + pool_width]
                indices = []
                for win in range(window.shape[0]):
                    arr = np.array([win]+list(np.unravel_index(np.argmax(window[win], axis=None), window[win].shape)))
                    indices.append(arr+[0, i * stride_height, j * stride_width])
                for index in indices:
                    output_array[tuple(index)] = output_grad[(index[0], i, j)]
        return output_array

class AvgPolling:
    def __init__(self, pool_size=(2, 2), strides=1) -> None:
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, data):
        self.data = data
        input_depth, input_height, input_width = data.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides, self.strides

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_array = np.zeros((input_depth, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                # Extract the region to pool
                window = data[:, i * stride_height: i * stride_height + pool_height,
                                    j * stride_width: j * stride_width + pool_width]
                output_array[:, i, j] = np.mean(window, axis=(1, 2))

        return output_array

    def backward(self, output_grad):
        output = np.zeros(self.data.shape)
        input_depth, input_height, input_width = data.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides, self.strides

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_array = np.zeros(self.data.shape)

        for i in range(output_height):
            for j in range(output_width):
                _grad = np.ones(tuple([output_grad.shape[0]])+self.pool_size)*(output_grad[:, i, j].reshape(-1, 1, 1)/4)
                output_array[:, i * stride_height: i * stride_height + pool_height,
                                    j * stride_width: j * stride_width + pool_width] = _grad
        return output_array
    
if __name__ == "__main__":
    conv = AvgPolling(strides=2)
    data = np.arange(128).reshape((2, 8, 8))
    print(data)
    print(conv.forward(data))
    grad = np.arange(32).reshape((2, 4, 4))
    print(conv.backward(grad))