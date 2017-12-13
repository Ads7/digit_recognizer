import os

from classifiers.logistic_regression import one_to_rest
from . import MODULE, BUFFER_SIZE
from utils.data_processing import write_results_to_file

BASE_NAME = os.path.basename(__file__).split('.')[0]

import time


def learning_rate_variation():
    """
    To study the variation of learning rate in the logistic regression
    the learning rate is varied from 0.1 to 0.9
    :return:  creates a csv with time taken , accuracy
    """
    results = []
    for i in range(1, 10,2):
        print "running for learning rate"+str(i)
        start_time = time.time()
        acc = one_to_rest(5000, i * 1.0 / 10.0)
        results.append([i, acc, time.time() - start_time])
        if len(results) % BUFFER_SIZE == 0:
            write_results_to_file(results, MODULE + BASE_NAME + ".csv")
            results = []
