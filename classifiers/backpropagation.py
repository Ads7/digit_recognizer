import random
import numpy as np

from utils.data_processing import DigitData
from utils.math import sigmoid

class Backpropagation(object):
    def __init__(self,learning_rate=3.5,iterations=10):
        self.learning_rate=learning_rate
        self.iterations=iterations
        (self.training_contents, self.test_contents) = self.load_data()
        self.main([784, 30, 10])

    def main(self, layers):
        self.num_layers = len(layers)
        self.feature_bias = [np.random.randn(layer, 1) for layer in layers[1:]]
        self.feature_weights = [np.random.randn(layer_two, layer_one)
                        for layer_one, layer_two in zip(layers[:-1], layers[1:])]
        self.gradient_descent()

    def feed_forward(self, image_test):
        input_values=zip(self.feature_bias, self.feature_weights)
        for bias, weight in input_values:
            image_test = sigmoid(np.dot(weight, image_test) + bias)
        return image_test

    def gradient_descent(self):
        for i in range(self.iterations):
            random.shuffle(self.training_contents)
            smaller_training_contents = [
                self.training_contents[j:j + 10]
                for j in range(0, len(self.training_contents), 10)]
            for small_tc in smaller_training_contents:
                self.tc_change_error(small_tc)
            print("Iteration", "[", i, "]", " : ", round(self.accuracy(self.test_contents), 2), "%")
            if (i == self.iterations-1):
                exit()
    def  get_delta_error(self,small_tc,update_b,update_w):
        for t_image, t_ans in small_tc:
            delta_b, delta_w = self.backpropagation_algo(t_image, t_ans)
            update_b = [update_bias + error_bias
                        for update_bias, error_bias
                        in zip(update_b, delta_b)]
            update_w = [update_weight + error_weight
                        for update_weight, error_weight
                        in zip(update_w, delta_w)]
        return update_b, update_w

    def feature_update(self,small_tc, update_b, update_w):
        training_set_length = len(small_tc)
        self.feature_weights = [weight - (self.learning_rate / training_set_length) * updated_weight
                                for weight, updated_weight in zip(self.feature_weights, update_w)]
        self.feature_bias = [bias - (self.learning_rate / training_set_length) * update_bias
                             for bias, update_bias in zip(self.feature_bias, update_b)]

    def tc_change_error(self, small_tc):
        update_b = [np.zeros(feature_bias.shape) for feature_bias in self.feature_bias]
        update_w = [np.zeros(feature_weight.shape) for feature_weight in self.feature_weights]
        update_b, update_w= self.get_delta_error(small_tc, update_b, update_w)
        self.feature_update( small_tc, update_b, update_w)

    def update_derivate_changes(self,derivative,changes,change):
        for feature_bias, feature_weight in zip(self.feature_bias, self.feature_weights):
            derivative_result = np.dot(feature_weight, change) + feature_bias
            derivative.append(derivative_result)
            change = sigmoid(derivative_result)
            changes.append(change)
        return derivative,changes,change

    def backpropagation_algo(self, t_image, t_ans):
        update_b = [np.zeros(feature_bias.shape) for feature_bias in self.feature_bias]
        update_w = [np.zeros(feature_weight.shape) for feature_weight in self.feature_weights]
        derivative, changes, change=self.update_derivate_changes([], [t_image], t_image)
        derivative_sig = sigmoid(derivative[-1])
        delta = (changes[-1] - t_ans) * (derivative_sig * (1 - derivative_sig))
        update_b[-1] = delta
        update_w[-1] = np.dot(delta, changes[-2].transpose())
        for layer_no in range(2, self.num_layers):
            derivative_result = derivative[-layer_no]
            derivative_result_sig = sigmoid(derivative_result)
            derivative_sig_prime = (derivative_result_sig * (1 - derivative_result_sig))
            delta = np.dot(self.feature_weights[-layer_no + 1].transpose(), delta) * derivative_sig_prime
            update_b[-layer_no] = delta
            update_w[-layer_no] = np.dot(delta, changes[-layer_no - 1].transpose())
        return (update_b, update_w)

    def accuracy(self, test_contents):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_contents]
        accurate_result = 0
        for (x, y) in test_results:
            if (int(x == y)):
                accurate_result += 1
        return (accurate_result / len(self.test_contents)) * 100

    def fill_array(self, ans):
        array = np.zeros((10, 1))
        array[ans] = 1.0
        return array

    def load_data(self):
        data = DigitData()
        train_images = np.asarray(data.mnist_data_bp.train.images)
        train_labels = np.asarray(data.mnist_data_bp.train.labels)
        test_images = np.asarray(data.mnist_data_bp.test.images)
        test_labels = np.asarray(data.mnist_data_bp.test.labels)
        training_inputs = [np.reshape(content, (784, 1)) for content in train_images]
        training_results = [self.fill_array(answer) for answer in train_labels]
        training_contents = zip(training_inputs, training_results)
        test_inputs = [np.reshape(content, (784, 1)) for content in test_images]
        test_contents = zip(test_inputs, test_labels)
        return (list(training_contents), list(test_contents))


