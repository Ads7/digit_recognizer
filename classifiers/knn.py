import numpy as np

from utils.data_processing import DigitData
from utils.math import euclidean_distance, calc_acc


class KNN(object):
    k = 10
    X_train = Y_train = None

    def __init__(self, X_train, Y_train, k=10):
        """
        :param X_train: the training data for the KNN classifier
        :type X_train: numpy array
        :param Y_train: labels from the training data
        :type Y_train: numpy array
        :param k: value of k(nearest neighbours)
        :type k:int
        """
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def get_nearest_neighbors(self, x):
        """

        :param x: single image array
        :return: nearest neighbors for given x
        :rtype: list
        """
        neighbor_distances = [(euclidean_distance(content, x), answer)
                              for (content, answer) in zip(self.X_train, self.Y_train)]
        neighbor_distance_sorted = sorted(neighbor_distances, key=lambda neighbor_distance: neighbor_distance[0])
        return neighbor_distance_sorted[:self.k]

    @staticmethod
    def get_majority(k_nearest_neighbors):
        """

        :param k_nearest_neighbors: nearest neighbours
        :return: majority
        """
        digit = np.zeros(10)
        index = np.argmax(np.array(np.array(k_nearest_neighbors)).sum(axis=0))
        digit[index] = 1
        return digit

    def get_prediction(self, x):
        """

        :param x: single X for testing
        :return: Y predictin for a given test data
        """
        neighbors_distance_list = self.get_nearest_neighbors(x)
        k_nearest_neighbors = [answer for (_, answer) in neighbors_distance_list]
        majority_vote_prediction = self.get_majority(k_nearest_neighbors)
        return majority_vote_prediction

    def predict(self, X_test):
        """
        :param X_test: Test data of images
        :return: numpy array of labels for the given X
        :type X_test: numpy array
        """
        result = []
        for i,x in enumerate(X_test):
            result.append(self.get_prediction(x))
        return np.array(result)
