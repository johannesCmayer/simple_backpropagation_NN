# -*- coding: utf-8 -*-
"""linear_classifier_iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zXpGPsCXUDmO6xiyprjhTrP-NV90pqEs
"""

import numpy as np
from sklearn.datasets import load_iris, make_blobs
import linear_classifier as lin_clf
import matplotlib.pyplot as plt

def shuffle(x, y):
    idx = np.random.permutation(x.shape[0])
    shuffeld_x = np.take(x, idx, 0)
    shuffeled_y = np.take(y, idx, 0)
    return shuffeld_x, shuffeled_y


def encode_int_onehot(int_to_encode, num_classes):
    vec = np.zeros(num_classes)
    vec[int_to_encode] = 1
    return vec


def get_data(id):
    data_get_func = {'iris': lambda: load_iris(return_X_y=True),
                     'blob': lambda: make_blobs(100, 2)}
    x_data, y_data = data_get_func.get(id)()
    x_data, y_data = shuffle(x_data, y_data)

    x_data_normalized = x_data / np.max(x_data)

    y_data = y_data.astype(np.int64)
    y_one_hot = np.array([encode_int_onehot(y, np.max(y_data) + 1) for y in y_data.astype(np.int64)])
    return x_data_normalized, y_one_hot


def run():
    x_data_normalized, y_one_hot = get_data('blob')
    y_int = np.argmax(y_one_hot, axis=-1)

    lc = lin_clf.LinearClassifier(x_data_normalized[0].size, y_one_hot[0].size)
    lc.train(x_data_normalized, y_one_hot, verbose=True, batch_size=None, epochs=600, learning_rate=0.01, loss=lin_clf.mse_loss)

    # grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    # grid = np.reshape(grid, (2, -1))
    # pred = lc.batch_eval(grid)
    plt.scatter([e[0] for e in x_data_normalized], [e[1] for e in x_data_normalized], c=y_int)

    for i in range(len(lc.weights)):
        plt.plot([-1,1], [-np.sum(lc.weights[i]), np.sum(lc.weights[i])])
    plt.show()


if __name__ == '__main__':
    run()
