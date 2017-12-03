import os

from classifiers.logistic_regression import one_to_rest
from . import MODULE, BUFFER_SIZE
from utils.data_processing import MAX_TRAIN_DATA, write_results_to_file

BASE_NAME = os.path.basename(__file__).split('.')[0]

import time


def variate_sample_size():
    results = []
    for i in range(500, MAX_TRAIN_DATA, 1000):
        start_time = time.time()
        acc = one_to_rest(i)
        results.append([i, acc, time.time() - start_time])
        if len(results) % BUFFER_SIZE == 0:
            write_results_to_file(results, MODULE + BASE_NAME + ".csv")
            results = []
