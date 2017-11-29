import numpy as np

from final_svm import DigitData


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in xrange(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            log_likelihood(features, target, weights)

    return weights


data = DigitData()
Y = np.array([1 if (y[1] == 1) else -1 for y in data.Ytrn])
weights = logistic_regression(data.Xtrn, Y,
                              num_steps=300000, learning_rate=5e-5, add_intercept=True)

X = data.Xtest.copy()
X = np.insert(X, 0, values=1, axis=1)

print sigmoid((X*weights).sum(axis=0))