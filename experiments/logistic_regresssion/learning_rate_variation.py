import os

from classifiers.logistic_regression import one_to_rest
from . import MODULE, BUFFER_SIZE
from utils.data_processing import MAX_TRAIN_DATA, write_results_to_file

BASE_NAME = os.path.basename(__file__).split('.')[0]

import time


def learning_rate_variation():
    results = []
    for i in range(1, 10):
        print "running for learning rate"+str(i)
        start_time = time.time()
        acc = one_to_rest(5000, i * 1.0 / 10.0)
        results.append([i, acc, time.time() - start_time])
        if len(results) % BUFFER_SIZE == 0:
            write_results_to_file(results, MODULE + BASE_NAME + ".csv")
            results = []
