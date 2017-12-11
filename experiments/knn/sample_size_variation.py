import time

from classifiers.knn import KNN
from experiments.logistic_regresssion.sample_size_variation import BASE_NAME
from . import BUFFER_SIZE, MODULE
from utils.data_processing import DigitData, write_results_to_file, MAX_TRAIN_DATA
from utils.math import calc_acc


def variate_sample_size():
    """
    To study the variation of training sample size from 500 to MAX_TRAIN_DATA step size as 1000
    :return: csv with acc. and time taken in each sample size
    """
    results = []
    for i in range(500, MAX_TRAIN_DATA, 1000):
        print "running for sample"+str(i)
        data = DigitData(MAX_TRAIN_DATA)
        start_time = time.time()
        model = KNN(k=i, X_train=data.X_train, Y_train=data.Y_train)
        y_predictions = model.predict(data.X_test)
        acc = calc_acc(data.Y_test, y_predictions)
        results.append([i, acc, time.time() - start_time])
        if len(results) % BUFFER_SIZE == 0:
            write_results_to_file(results, MODULE + BASE_NAME + ".csv")
            results = []
