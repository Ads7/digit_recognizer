import numpy  as np
import random
# import matplotlib.pyplot as plt

from final_svm import DigitData


class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, max_iter=500, C=1.0):
        self.max_iter = max_iter
        self.C = C

    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2 * self.kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            # if diff < 0.01:
            #     break
            if count >= self.max_iter:
                # print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                break
            # # Check convergence
            # diff = np.linalg.norm(alpha - alpha_prev)
            # if diff < self.epsilon:
            #     break


        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        # if self.kernel_type == 'linear':
        self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)
    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = random.randint(a,b)
            cnt=cnt+1
        return i
    # Define kernels
    def kernel(self, x1, x2):
        return np.dot(x1, x2.T)


def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

model = SVM()

data = DigitData()
predictive_model = np.full((data.Ytest.shape[0], 10), 0, dtype=int)
confidence_list = []
for clasification in data.classses:
    print "model for "+ str(clasification)
    Y = np.array([1 if (y[clasification] == 1) else -1 for y in data.Ytrn])
    # Fit model
    support_vectors, iterations = model.fit(data.Xtrn, Y)
    y_hat = model.predict(data.Xtest)
    for i,y in enumerate(y_hat):
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
random_val  = 0
for i in range(1000):
    value = np.where(data.Ytest[i] == 1)[0][0]
    value_predict = np.where(predictive_model[i] == 1)[0]
    if len(np.where(predictive_model[i] == 1)[0]) > 1:
        value_predict = confidence_list.index(max(map(lambda x:confidence_list[x], value_predict)))
    # print np.where(data.Ytest[i] == 1) == np.where(predictive_model[i] == 1)
    # print len(np.where(predictive_model[i] == 1)[0])
    # print len(np.where(predictive_model[i] == 1))
    # print predictive_model[i]
    if value_predict == value:
        count +=1
    else:
        print value, value_predict
print count
print random_val
# Calculate accuracy
# acc = calc_acc(data.Ytest, predictive_model)

# print("Support vector count: %d" % (sv_count))
# print("bias:\t\t%.3f" % (model.b))
# print("w:\t\t" + str(model.w))
# print("accuracy:\t%.3f" % (acc))
# print("Converged after %d iterations" % (iterations))
