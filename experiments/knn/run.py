import time

from classifiers.knn import KNN
from utils.data_processing import DigitData
from utils.math import calc_acc


def run():
    """
    To run Knn with sample size 5000
    :return: prints the acc and time taken
    """
    data = DigitData(5000)
    start_time = time.time()
    model = KNN(X_train=data.X_train, Y_train=data.Y_train)
    y_predictions = model.predict(data.X_test)
    acc = calc_acc(data.Y_test, y_predictions)
    print ([acc, time.time() - start_time])
