# Cole Chamberlin, Ishan Saksena
# mnist dataset
# CSE415, Final Project
# Spring 2017

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
    return training_data, training_labels, testing_data, testing_labels
