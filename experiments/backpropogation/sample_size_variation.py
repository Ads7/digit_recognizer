import time

from classifiers.knn import KNN
from experiments.logistic_regresssion.sample_size_variation import BASE_NAME
from . import BUFFER_SIZE, MODULE
from utils.data_processing import DigitData, write_results_to_file, MAX_TRAIN_DATA
from utils.math import calc_acc


