import numpy  as np
import random
# import matplotlib.pyplot as plt

from final_svm import DigitData


class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(self, max_passes=500, C=1.0, tol=0.01):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        # Initialization
        m = X.shape[0]  # training sample size
        b = 0
        alpha = np.zeros((m))  # Lagranges multipliers
        passes = 0
        while passes < self.max_passes:
            num_changed_alpha = 0
            for i in range(0, m):
                x_i, y_i = X[i, :], y[i]
                alpha_i = alpha[i]
                E_i = self.f(x=x_i, alpha=alpha, X=X, Y=y, b=b, m=m) - y_i
                if (y_i * E_i < -self.tol and alpha_i < self.C) or (y_i * E_i > self.tol and alpha_i > 0):
                    j = self.get_rnd(m - 1, i)  # Get random int j~=i
                    x_j, y_j = X[j, :], y[j]
                    E_j = self.f(x=x_j, alpha=alpha, X=X, Y=y, b=b, m=m)-y_j
                    alpha_j = alpha_j_old = alpha[j]
                    alpha_i_old = alpha_i
                    L, H = self.compute_L_H(self.C, alpha_i, alpha_j, y_i, y_j)
                    if L == H:
                        continue
                    eta = 2 * self.kernel(x_i, x_j) - self.kernel(x_j, x_j) - self.kernel(x_i, x_i)
                    if eta >= 0:
                        continue
                    # compute alpha_j
                    alpha_j = alpha_j + float(y_j * (E_i - E_j)) / eta
                    # assign too
                    # clip alpha_j
                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L
                    alpha[j] = alpha_j
                    if np.linalg.norm(alpha_j - alpha_j_old) < 0.00001:
                        continue
                    alpha_i = alpha_i + y_i * y_j * (alpha_j_old - alpha_j)
                    alpha[i] = alpha_i
                    b1 = b2 = 0
                    if 0 < alpha_i < self.C:
                        b1 = b - E_i - y_i * (alpha_i - alpha_i_old) * self.kernel(x_i, x_i) - y_j * (
                            alpha_j - alpha_j_old) * self.kernel(x_i, x_j)
                    if 0 < alpha_j < self.C:
                        b2 = b - E_j - y_i * (alpha_i - alpha_i_old) * self.kernel(x_i, x_j) - y_j * (
                            alpha_j - alpha_j_old) * self.kernel(x_j, x_j)
                    b = (b1 + b2) / 2
                    num_changed_alpha += 1
            if num_changed_alpha == 0:
                passes += 1
            else:
                passes = 0

        # Compute final model parameters
        self.b = b
        # if self.kernel_type == 'linear':
        self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, passes

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        # weight = np.zeros((data.Xtrn.shape[1], 1), dtype=np.float32).T
        #
        # for i in range(data.Ytrn.shape[0]):
        #     weight += alpha[i] * Y[i] * data.Xtrn[i]
        return np.dot(alpha * y, X)

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    # def E(self, x_k, y_k, w, b):
    #     return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            return max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i)
        else:
            return max(0, alpha_i + alpha_j - C), min(C, alpha_j + alpha_i)

    def get_rnd(self, limit, val):
        j = val
        while j == val:
            j = random.randint(0, limit)
        return j

    # Define kernels
    def kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def f(self, x, alpha, X, Y, b, m):
        sum = 0
        for i in range(m):
            sum = sum + (alpha[i] * Y[i] * self.kernel(X[i], x))
        sum += b
        return sum


def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN) / len(y)


model = SVM()

data = DigitData()
predictive_model = np.full((data.Ytest.shape[0], 10), 0, dtype=int)
confidence_list = []
for clasification in data.classses:
    print "model for " + str(clasification)
    Y = np.array([1 if (y[clasification] == 1) else -1 for y in data.Ytrn])
    # Fit model
    support_vectors, iterations = model.fit(data.Xtrn, Y)
    y_hat = model.predict(data.Xtest)
    for i, y in enumerate(y_hat):
        if y == -1:
            pass
        else:
            predictive_model[i][clasification] += y
    ytest = np.array([1 if (y[clasification] == 1) else -1 for y in data.Ytest])
    confidence_list.append(calc_acc(ytest, y_hat))
# from sklearn import datasets, svm, metrics
# # The digits dataset
# digits = datasets.load_digits()
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# X = data.Xtest[1].reshape([28, 28])
# plt.gray()
# plt.imshow(X)


# Make prediction
# print predictive_model[0]
count = 0
random_val = 0
for i in range(10):
    value = np.where(data.Ytest[i] == 1)[0][0]
    value_predict = np.where(predictive_model[i] == 1)[0]
    if len(np.where(predictive_model[i] == 1)[0]) > 1:
        value_predict = confidence_list.index(max(map(lambda x: confidence_list[x], value_predict)))
    # print np.where(data.Ytest[i] == 1) == np.where(predictive_model[i] == 1)
    # print len(np.where(predictive_model[i] == 1)[0])
    # print len(np.where(predictive_model[i] == 1))
    # print predictive_model[i]
    if value_predict == value:
        count += 1
    else:
        print value, value_predict
print count
print random_val
# Calculate accuracy
acc = calc_acc(data.Ytest, predictive_model)

# print("Support vector count: %d" % (sv_count))
# print("bias:\t\t%.3f" % (model.b))
# print("w:\t\t" + str(model.w))
print("accuracy:\t%.3f" % (acc))
# print("Converged after %d iterations" % (iterations))
