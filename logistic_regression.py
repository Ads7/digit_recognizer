import numpy as np

from final_svm import DigitData
from mlfromscratch.supervised_learning.logistic_regression import LogisticRegression


class LogisticRegression():
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    """

    @staticmethod
    def make_diagonal(x):
        """ Converts a vector into an diagonal matrix """
        m = np.zeros((len(x), len(x)))
        for i in range(len(m[0])):
            m[i, i] = x[i]
        return m

    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        from mlfromscratch.deep_learning.activation_functions import Sigmoid
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = self.make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param)))
        return y_pred.astype(int)

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN) / len(y)

data = DigitData()
predictive_model = np.full((data.Ytest.shape[0], 10), 0, dtype=int)
confidence_list = []
for clasification in data.classses:
    print "model for " + str(clasification)
    Y = np.array([1 if (y[clasification] == 1) else 0 for y in data.Ytrn])
    # Fit model
    model = LogisticRegression()
    model.fit(data.Xtrn, Y)
    y_hat = model.predict(data.Xtest)
    for i, y in enumerate(y_hat):
        if y == 1:
            predictive_model[i][clasification] += y
    ytest = np.array([1 if (y[clasification] == 1) else 0 for y in data.Ytest])
    confidence_list.append(calc_acc(ytest, y_hat))

count = 0
random_val = 0
for i in range(10):
    value = np.where(data.Ytest[i] == 1)[0][0]
    value_predict = np.where(predictive_model[i] == 1)[0]
    if len(np.where(predictive_model[i] == 1)[0]) > 1:
        value_predict = confidence_list.index(max(map(lambda x: confidence_list[x], value_predict)))
    if value_predict == value:
        count += 1
    else:
        print value, value_predict
print count
print random_val
# Calculate accuracy
acc = calc_acc(data.Ytest, predictive_model)
print("accuracy:\t%.3f" % (acc))
