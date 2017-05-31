# Cole Chamberlin, Ishan Saksena
# mnist dataset
# CSE415, Final Project
# Spring 2017

# data taken from http://yann.lecun.com/exdb/mnist/

from mnist import MNIST
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt
import PCA

def get_data():
    #nbc.DISP_MNIST = True
    mndata = MNIST('samples')
    training_data, training_labels = mndata.load_training()
    testing_data, testing_labels = mndata.load_testing()
    return np.array(training_data), np.array(training_labels), \
            np.array(testing_data), np.array(testing_labels)

if __name__ == '__main__':
    training_data, training_labels, testing_data, testing_labels = get_data()
    n = nbc()
    n.reporting = True
    # include these two lines to see visual display of digits
    n.disp_mnist = True
    n.images = testing_data

    if str(type(n)) == "<class 'NaiveBayesClassifier.NaiveBayesClassifier'>":
        training_data = PCA.reduceDim(training_data, 28)
        testing_data = PCA.reduceDim(testing_data)

    n.train(training_data, training_labels)
    n.test(testing_data, testing_labels)
