import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import json
import os
import logging

class DataGetter:
    def __init__(self, data_location="D:/Data/"):
        self.data_location = data_location

    def get_data(self, data_name, cache_data=True, **kwargs):
        data_path = self.data_location + data_name + '.json'
        data = None
        if os.path.isfile(data_path) and os.stat(data_path).st_size == 0:
            os.remove(data_path)
        if os.path.isfile(data_path):
            print('Loading cached data')
            with open(data_path, 'r') as f:
                data = json.loads(f.read())
        else:
            print('Downloading data')
            data = fetch_openml(data_name, **kwargs)
            if cache_data:
                with open(data_path, 'w') as f:
                    def convert_ndarray(v):
                        return v.tolist() if type(v).__module__ == np.__name__ else v
                    json_data = {k:convert_ndarray(v) for (k, v) in zip(data.keys(), data.values())}
                    f.write(json.dumps(json_data))
        print('load finished')
        return np.array(data)


x_data, y_data = fetch_openml('mnist_784', version=1, return_X_y=True)


def hinge_loss(x, y):
    return np.max(0, x - y)


class LinearClassifier():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)

    def eval(self, inputs):
        result = np.matmul(self.weights, inputs)
        unnormal_probabilities = np.exp(result)
        probabilities = unnormal_probabilities / sum(unnormal_probabilities)
        return probabilities

    def cost(self):
        pass

    def train(self):
        pass


def run():
    input_size = len(x_data[0])
    number_of_labels = len(y_data[0])
    lc = LinearClassifier(input_size, 10)
    print(np.isnan(x_data[0]).any())
    print(not np.isfinite(x_data[0]).any())
    test_data = x_data[1] * 0.1 + 0.1
    print(test_data)
    print(lc.eval(test_data))

run()

image_side_length = int(np.sqrt(len(x_data[0])))
for i in range(1):
    plt.imshow(x_data[i].reshape(image_side_length, image_side_length))
    plt.show()