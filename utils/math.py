import numpy as np

def euclidean_distance(x1,x2):
    distance = sum((x1 - x2) ** 2)
    return distance

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

def calc_acc(y, y_prediction):
    idx = np.where(y_prediction == 1)
    TP = np.sum(y_prediction[idx] == y[idx])
    return float(TP) / len(y)