import numpy as np

from utils.data_processing import DigitData
from utils.math import sigmoid, calc_acc


class LogisticRegression(object):
    def __init__(self, eta):
        self.param = None
        self.eta = eta

    def _initialize_parameters(self, X):
        feature_size = X.shape[1]
        limit = 1 / np.math.sqrt(feature_size)
        self.param = np.random.uniform(-limit, limit, (feature_size,))

    def gradient_decent(self,X,y,y_pred):
        self.param -= self.eta * -(y - y_pred).dot(X)

    def fit(self, X, y, epoch=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(epoch):
            # Make a new prediction
            y_pred = sigmoid(X.dot(self.param))
            # minimize the loss
            self.gradient_decent(X,y,y_pred)


    def predict(self, X):
        return np.round(sigmoid(X.dot(self.param))).astype(int)


def one_to_rest(sample_size=5000,learning_rate=0.1):
    data = DigitData(sample_size)
    predictive_model = np.full((data.Y_test.shape[0], 10), 0, dtype=int)
    confidence_list = []
    model = LogisticRegression(learning_rate)
    for clasification in data.classes:
        print "model for " + str(clasification)
        Y = np.array([1 if (y[clasification] == 1) else 0 for y in data.Y_train])
        # Fit model
        model.fit(data.X_train, Y)
        y_hat = model.predict(data.X_test)
        for i, y in enumerate(y_hat):
            if y == 1:
                predictive_model[i][clasification] += y
        ytest = np.array([1 if (y[clasification] == 1) else 0 for y in data.Y_test])
        confidence_list.append(calc_acc(ytest, y_hat))

    for i in range(data.Y_test.shape[0]):
        value_predict = np.where(predictive_model[i] == 1)[0]
        if len(np.where(predictive_model[i] == 1)[0]) > 1:
            digit = np.zeros(10)
            value_predict = confidence_list.index(max(map(lambda x: confidence_list[x], value_predict)))
            digit[value_predict] = 1
            predictive_model[i] = digit
    # Calculate accuracy
    return calc_acc(data.Y_test, predictive_model)

