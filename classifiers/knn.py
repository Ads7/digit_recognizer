import numpy as np

from utils.data_processing import DigitData
from utils.math import euclidean_distance, calc_acc


class KNN(object):
    k = 10
    X_train = Y_train = None

    def __init__(self, X_train, Y_train, k=10):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def get_nearest_neighbors(self, test_content):
        neighbor_distances = [(euclidean_distance(content, test_content), answer)
                              for (content, answer) in zip(self.X_train, self.Y_train)]
        neighbor_distance_sorted = sorted(neighbor_distances, key=lambda neighbor_distance: neighbor_distance[0])
        return neighbor_distance_sorted[:self.k]

    def get_majority(self, k_nearest_neighbors):
        digit = np.zeros(10)
        index = np.argmax(np.array(np.array(k_nearest_neighbors)).sum(axis=0))
        digit[index] = 1
        return digit

    def get_prediction(self, test_content):
        neighbors_distance_list = self.get_nearest_neighbors(test_content)
        k_nearest_neighbors = [answer for (_, answer) in neighbors_distance_list]
        majority_vote_prediction = self.get_majority(k_nearest_neighbors)
        return majority_vote_prediction

    def predict(self, X_test):
        return np.array(map(lambda x: self.get_prediction(x), X_test))