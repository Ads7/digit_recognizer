import csv

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

ROOT_DIR = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIGIT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_TRAIN_DATA = 55000
MAX_TEST_DATA = 10000


class DigitData(object):
    mnist = input_data.read_data_sets(ROOT_DIR + "/data/", one_hot=True)
    mnist_data_bp = input_data.read_data_sets(ROOT_DIR + "/data/", one_hot=False)
    classes = DIGIT_CLASSES
    X_train, Y_train = np.array(mnist.train.images), np.array(mnist.train.labels, dtype=int)
    X_test, Y_test = np.array(mnist.test.images), np.array(mnist.test.labels, dtype=int)


    def __init__(self, train_limit=None, test_limit=None):
        if train_limit and train_limit < MAX_TRAIN_DATA:
            self.X_train, self.Y_train = self.X_train[:train_limit], self.Y_train[:train_limit]
        if test_limit and test_limit < MAX_TEST_DATA:
            self.X_test, self.Y_test = self.X_test[:test_limit], self.Y_test[:test_limit]


def write_results_to_file(values, filename):
    with open(ROOT_DIR + '/results/' + filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(values)
