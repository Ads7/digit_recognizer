import numpy as np

from utils.data_processing import DigitData
from utils.math import sigmoid, calc_acc


class LogisticRegression(object):
    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate

    def _initialize_parameters(self, X):
        n_features = X.shape[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = sigmoid(X.dot(self.param))
            # Move against the gradient of the loss function with
            # respect to the parameters to minimize the loss
            self.param -= self.learning_rate * -(y - y_pred).dot(X)

    def predict(self, X):
        y_pred = np.round(sigmoid(X.dot(self.param)))
        return y_pred.astype(int)


def one_to_rest(sample_size=1000):
    data = DigitData(sample_size)
    predictive_model = np.full((data.Y_test.shape[0], 10), 0, dtype=int)
    confidence_list = []
    model = LogisticRegression()
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
    calc_acc(data.Y_test, predictive_model)

