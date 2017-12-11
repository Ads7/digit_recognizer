import os

from classifiers.logistic_regression import one_to_rest
from . import MODULE, BUFFER_SIZE
from utils.data_processing import MAX_TRAIN_DATA, write_results_to_file

BASE_NAME = os.path.basename(__file__).split('.')[0]

import time


def variate_sample_size():
    """
    To study the variation of sample size in logistic regression
    the variation are form 500 to MAX_TRAIN_DATA with step size of 1000
    :return: creates a csv of observations
    """
    results = []
    for i in range(500, MAX_TRAIN_DATA, 1000):
        print "running for sample"+str(i)
        start_time = time.time()
        acc = one_to_rest(i)
        results.append([i, acc, time.time() - start_time])
        if len(results) % BUFFER_SIZE == 0:
            write_results_to_file(results, MODULE + BASE_NAME + ".csv")
            results = []
