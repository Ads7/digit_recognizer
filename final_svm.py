import os
import random

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
CWD = os.getcwd()
from sklearn.svm import SVC


class DigitData(object):
    # Step 1 - Define our data
    mnist = input_data.read_data_sets(CWD + "/data/", one_hot=True)
    classses = [0,1,2,3,4,5,6,7,8,9]
    # todo remove limit
    Xtrn, Ytrn = np.array(mnist.train.images[:10000]), np.array(mnist.train.labels[:10000],dtype=int)
    Xtest, Ytest = np.array(mnist.test.images), np.array(mnist.test.labels,dtype=int)



def determineB1(yTrnI, yTrnJ, alphaI, alphaJ, prevAlphaI, prevAlphaJ, xTrnI, xTrnJ):
    prod1 = yTrnI * (alphaI - prevAlphaI) * kernel(xTrnI, xTrnI, 1)
    prod2 = yTrnJ * (alphaJ - prevAlphaJ) * kernel(xTrnJ, xTrnI, 1)
    return prod1 + prod2

def determineB2(yTrnI, yTrnJ, alphaI, alphaJ, prevAlphaI, prevAlphaJ, xTrnI, xTrnJ):
    prod1 = yTrnI * (alphaI - prevAlphaI) * kernel(xTrnJ, xTrnI, 1)
    prod2 = yTrnJ * (alphaJ - prevAlphaJ) * kernel(xTrnJ, xTrnJ.T, 1)
    return prod1 + prod2

def computeAndClip(alpha, Ytrn, Ei, Ej, eta):
    change = (Ytrn*(Ei - Ej)) / eta
    return alpha - change

def calculateEta(Xi, Xj):
    temp = 2*kernel(Xi, Xj, 1)
    temp -= kernel(Xi, Xi, 1)
    return temp - kernel(Xj.T, Xj, 1)

def computeClassifier(Xj, b, alpha,X,Y):
    sum = 0
    for i in range(Y.shape[0]):
        dotProd = kernel(X[i], Xj, 1)
        sum = sum + (alpha[i] * Y[i] * dotProd)
    sum += b
    return sum

def kernel(X, Z, maxDegree):
    prod = np.dot(X, Z)
    return prod**maxDegree

def calculateLH(yI, yJ, regParam, alphaJ, alphaI):
    if (yI != yJ):
        l = max(0, alphaJ - alphaI)
        h = min(regParam, regParam + alphaJ - alphaI)
    else:
        l = max(0, alphaJ + alphaI - regParam)
        h = min(regParam, alphaJ + alphaI)
    return l, h

def determineAlpha(alphaI, yTrnI, yTrnJ, prevAlphaJ, alphaJ):
    prod = yTrnI*yTrnJ
    prod = prod * (prevAlphaJ - alphaJ)
    return alphaI + prod


class SVM(object):
    tolrence = 0.01
    # how many iterations to train for
    epochs = 1000
    def train(self):
        data = DigitData()
        Y = np.array([1 if (y[0] == 1) else 0 for y in data.Ytrn])
        # self.trial_svm(self.epochs,data.Xtrn,Y)

        # alpha, b, count = self.simplified_smo(1, self.epochs, data.Xtrn, data.Ytrn, self.tolrence)
        # w = self.get_weight(data.Xtrn,data.Ytrn,alpha)
        # Y = self.svm(data.Xtrn,w , b)

    def get_weight(self, X, Y,alpha):
        weight = np.zeros((X.shape[1], 1), dtype=np.float32).T
        for i in range(Y.shape[0]):
            weight += alpha[i] * Y[i] * X[i]
        return weight

    def svm(self,X,w,b):
        op = []
        for i in range(X.shape[0]):
            ops = np.dot(X[i, :], w.T) + b
            op.append(ops)
        return op

    def trial_svm(self,epochs,X,Y):
        # lets perform stochastic gradient descent to learn the seperating hyperplane between both classes
        # Initialize our SVMs weight vector with zeros (3 values)
        w = np.random.randint(2, size=X.shape[1])
        # The learning rate
        eta = 1
        # store misclassifications so we can plot how they change over time
        errors = []

        # training part, gradient descent part
        for epoch in range(1, epochs):
            for i in range(X.shape[0]):
                # misclassification
                if (Y[i] * np.dot(X[i], w)) < 1:
                    # misclassified update for ours weights
                    w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
                else:
                    # correct classification, update our weights
                    w = w + eta * (-2 * (1 / epoch) * w)
        return w

    def simplified_smo(self, reg_param, iterations, X, Y, tol):
        # http://cs229.stanford.edu/materials/smo.pdf
        alpha = np.zeros(Y.shape)
        prevAlpha = np.zeros(Y.shape)
        b = 0
        passes = 0
        while passes < iterations:
            modifiedCount = 0
            print(passes)
            for i in range(Y.shape[0]):
                fi = computeClassifier(X[i], b, alpha,X,Y)
                Ei = fi - Y[i]
                if (Y[i] * Ei < -tol and alpha[i] < reg_param) or (Y[i] * Ei > tol and alpha[i] > 0):
                    while True:
                        j = random.sample(range(0, Y.shape[0] - 1), 1)[0]

                        if (i != j): break
                    fj = computeClassifier(X[j], b, alpha,X,Y)
                    Ej = fj - Y[j]
                    prevAlpha[i] = alpha[i]
                    prevAlpha[j] = alpha[j]

                    L, H = calculateLH(Y[i], Y[j], reg_param, alpha[j], alpha[i])
                    if (L == H): continue

                    eta = calculateEta(X[i], X[j].T)
                    if (eta >= 0): continue

                    alpha[j] = computeAndClip(alpha[j], Y[j], Ei, Ej, eta)
                    if (alpha[j] > H): alpha[j] = H
                    if (alpha[j] < L): alpha[j] = L

                    if (abs(alpha[j] - prevAlpha[j]) < 10 ** -5): continue

                    alpha[i] = determineAlpha(alpha[i], Y[i], Y[j], prevAlpha[j], alpha[j])

                    b1 = b - Ei - determineB1(Y[i], Y[j], alpha[i], alpha[j], prevAlpha[i], prevAlpha[j], X[i],
                                              X[j])
                    b2 = b - Ej - determineB2(Y[i], Y[j], alpha[i], alpha[j], prevAlpha[i], prevAlpha[j], X[i],
                                              X[j])

                    if (alpha[i] < reg_param and alpha[i] > 0):
                        b = b1
                    elif (alpha[j] < reg_param and alpha[j] > 0):
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    modifiedCount += 1
            if (modifiedCount == 0):
                passes += 1
            else:
                passes = 0

        return alpha, b, modifiedCount
#
# SVM().train()