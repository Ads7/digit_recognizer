import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from utils.data_processing import DigitData
from utils.math import sigmoid

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=False)


class Backpropagation(object):
    def __init__(self,learning_rate=3.5,iterations=10):
        self.learning_rate=learning_rate
        self.iterations=iterations
        (self.training_contents, self.test_contents) = self.load_data()
        self.main([784, 30, 10])

    def main(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]
        self.gradient_descent()

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self):
        for i in range(self.iterations):
            random.shuffle(self.training_contents)
            smaller_training_contents = [
                self.training_contents[j:j + 10]
                for j in range(0, len(self.training_contents), 10)]
            for small_tc in smaller_training_contents:
                self.tc_change_error(small_tc)
            print("Iteration", "[", i, "]", " : ", round(self.accuracy(self.test_contents), 2), "%")
            if (i == 9):
                exit()

    def tc_change_error(self, small_tc):
        update_b = [np.zeros(b.shape) for b in self.biases]
        update_w = [np.zeros(w.shape) for w in self.weights]
        training_set_length = len(small_tc)
        for x, y in small_tc:
            delta_b, delta_w = self.backpropagation_algo(x, y)
            update_b = [update_bias + error_bias
                        for update_bias, error_bias
                        in zip(update_b, delta_b)]
            update_w = [update_weight + error_weight
                        for update_weight, error_weight
                        in zip(update_w, delta_w)]
        self.weights = [weight - (self.learning_rate / training_set_length) * updated_weight
                        for weight, updated_weight in zip(self.weights, update_w)]
        self.biases = [bias - (self.learning_rate / training_set_length) * update_bias
                       for bias, update_bias in zip(self.biases, update_b)]

    def backpropagation_algo(self, x, y):
        update_b = [np.zeros(b.shape) for b in self.biases]
        update_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        zs_sig = sigmoid(zs[-1])
        delta = (activations[-1] - y) * (zs_sig * (1 - zs_sig))
        update_b[-1] = delta
        update_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            z_sig = sigmoid(z)
            sp = (z_sig * (1 - z_sig))
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            update_b[-l] = delta
            update_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (update_b, update_w)

    def accuracy(self, test_contents):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_contents]
        accurate_result = 0
        for (x, y) in test_results:
            if (int(x == y)):
                accurate_result += 1
        return (accurate_result / len(self.test_contents)) * 100

    def fill_array(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def load_data(self):
        data = DigitData()
        training_inputs = [np.reshape(content, (784, 1)) for content in data.X_train]
        training_results = [self.fill_array(answer) for answer in data.Y_train]
        training_contents = zip(training_inputs, training_results)
        test_inputs = [np.reshape(content, (784, 1)) for content in data.X_test]
        test_contents = zip(test_inputs, data.Y_test)

        return (list(training_contents), list(test_contents))


bp = Backpropagation()
bp.main([784, 100, 10])
