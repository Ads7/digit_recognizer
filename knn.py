import numpy as numpy
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.examples.tutorials.mnist import input_data
mnist_data=input_data.read_data_sets("MNIST_data/",one_hot=False)

training_contents = numpy.asarray(mnist_data.train.images[:5000])
training_answers = numpy.asarray(mnist_data.train.labels[:5000])
test_contents = numpy.asarray(mnist_data.test.images)
test_answers = numpy.asarray(mnist_data.test.labels)


def euclidean_distance(image_one,image_two):
    distance = sum((image_two - image_one) ** 2)
    return distance

def get_neighbor_distance(test_content):
    neighbor_distances = [(euclidean_distance(content,test_content),answer)
                          for (content, answer) in zip(training_contents, training_answers)]
    neighbor_distance_sorted=sorted(neighbor_distances,key=lambda neighbor_distance:neighbor_distance[0])
    return neighbor_distance_sorted

def get_majority(k_nearest_neighbors):
    majority=defaultdict(int)
    for neighbor in k_nearest_neighbors:
        majority[neighbor]+=1
    max_vote=max(majority.values())
    for key,value in majority.items():
        if value==max_vote:
            return key


def get_prediction (test_content,k):
    neighbors=get_neighbor_distance(test_content)
    k_nearest_neighbors=[answer for (_,answer) in neighbors[:k]]
    majority_vote_prediction=get_majority(k_nearest_neighbors)
    return majority_vote_prediction

def get_accuracy(no_of_test,no_of_accurate_test):
    accuracy_percentage= (no_of_accurate_test/no_of_test)*100
    accuracy_percentage= (round (accuracy_percentage, 2))
    return accuracy_percentage

def main():
    accurate_prediction=0
    i=0
    for test_content in test_contents:
        test_content_result = get_prediction(test_content,10)
        if test_content_result==test_answers[i]:
            accurate_prediction+=1
        accuracy=get_accuracy(i+1,accurate_prediction)
        X = test_content.reshape([28, 28])
        plt.gray()
        plt.imshow(X)
        #print (X)
        #plt.show()
        print('\ntest image index = ', str(i))
        print('prediction result = ', test_content_result, '\tanswer =', test_answers[i])
        print ('prediction correct? =' ,test_content_result==test_answers[i])
        print('prediction accuracy=', str(accuracy) + '%')
        i+=1
main()