# Cole Chamberlin, Ishan Saksena
# breast cancer dataset
# CSE415, Final Project
# Spring 2017

# data taken from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

import pandas as pd
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt
import PCA
import Boosting

def get_data():
    data = pd.read_csv("data.csv")
    labels = np.array(data.ix[1:,1])
    data = np.array(data.ix[1:,2:32])

    return PCA.trainTestSplit(data,labels,.8)

if __name__ == '__main__':
    training_data, testing_data, training_labels, testing_labels = get_data()
    # Decision Tree classifier
    c1 = dt()
    c1.train(training_data, training_labels)
    c1.test(testing_data, testing_labels)
    # Naive Bayes Classifier
    c2 = nbc()
    PCA.reduceDim(training_data, 12)
    PCA.reduceDim(testing_data)
    c2.train(training_data, training_labels)
    c2.test(testing_data, testing_labels)


