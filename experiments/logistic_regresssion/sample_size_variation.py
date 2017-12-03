import csv

from classifiers.logistic_regression import one_to_rest
from utils.data_processing import MAX_TRAIN_DATA


def write_to_file(values):
    with open('mycsvfile.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, 0, 0])
        writer.writerow([0, 0, 0])
        writer.writerow([0, 0, 0])

def variate_sample_size():
    for i in range(0,MAX_TRAIN_DATA,5000):
        acc = one_to_rest(i)