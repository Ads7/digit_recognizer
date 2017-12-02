import os

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

CWD = os.getcwd()

DIGIT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_TRAIN_DATA = 0
MAX_TEST_DATA = 0


class DigitData(object):
    mnist = input_data.read_data_sets(CWD + "/data/", one_hot=True)
    classes = DIGIT_CLASSES
    X_train, Y_train = np.array(mnist.train.images), np.array(mnist.train.labels, dtype=int)
    X_test, Y_test = np.array(mnist.test.images), np.array(mnist.test.labels, dtype=int)

    def __init__(self, train_limit=None, test_limit=None):
        if train_limit and train_limit < MAX_TRAIN_DATA:
            self.X_train, self.Y_train = self.X_train[:train_limit], self.Y_train[:train_limit]
        if test_limit and test_limit < MAX_TEST_DATA:
            self.X_test, self.Y_test = self.X_test[:test_limit], self.Y_test[:test_limit]

    @staticmethod
    def calc_acc(y, y_hat):
        idx = np.where(y_hat == 1)
        TP = np.sum(y_hat[idx] == y[idx])
        idx = np.where(y_hat == -1)
        TN = np.sum(y_hat[idx] == y[idx])
        return float(TP + TN) / len(y)

    def accuracy(self, model):
        test_size = self.Y_test.shape[0]
        predictive_model = np.full((test_size, 10), 0, dtype=int)
        confidence_list = []
        for clasification in self.classes:
            print "model for " + str(clasification)
            Y = np.array([1 if (y[clasification] == 1) else 0 for y in self.Y_train])
            # Fit model
            model.fit(self.X_train, Y)
            y_hat = model.predict(self.X_test)
            for i, y in enumerate(y_hat):
                if y == 1:
                    predictive_model[i][clasification] += y
            ytest = np.array([1 if (y[clasification] == 1) else 0 for y in self.Y_test])
            confidence_list.append(self.calc_acc(ytest, y_hat))
        count = 0
        for i in range(test_size):
            value = np.where(self.Y_test[i] == 1)[0][0]
            value_predict = np.where(predictive_model[i] == 1)[0]
            # todo update predictive model
            if len(np.where(predictive_model[i] == 1)[0]) > 1:
                confidence_list.index(max(map(lambda x: confidence_list[x], value_predict)))
            if value_predict == value:
                count += 1
            else:
                print value, value_predict
        acc = count/test_size

        acc = self.calc_acc(self.Y_test, predictive_model)