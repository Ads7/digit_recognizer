import numpy as np


def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))