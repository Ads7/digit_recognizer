import time

from classifiers.knn import KNN
from classifiers.logistic_regression import one_to_rest
from utils.data_processing import DigitData
from utils.math import calc_acc


def run():
    start_time = time.time()
    acc = one_to_rest(5000)
    print ([acc, acc, time.time() - start_time])
