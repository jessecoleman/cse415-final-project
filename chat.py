# Cole Chamberlin, Ishan Saksena
# Naive Bayes Classifier
# CSE415, Final Project
# Spring 2017

from NaiveBayesClassifier import NaiveBayesClassifier as nb
from DecisionTree import DecisionTree as dt
import mnist_data
import BreastCancer
import PCA
import Boosting

if __name__ == "__main__":
    # Parameters for classification
    dataset_choice = input("What dataset do you want to use? [Breast Cancer (b) / Handwritten digits (h)] ")
    if dataset_choice == "b":
        training_data, training_labels, testing_data, testing_labels = BreastCancer.get_data()
    else:
        training_data, training_labels, testing_data, testing_labels = mnist_data.get_data()

    classifier_choice = input("What kind of classifier do you want to use? [Naive bayes (n) / Decision Tree (d)] ")
    if classifier_choice == "n":
        classifier = nb()
    else:
        classifier = dt()

    print("The current dimensions of the data set are:", training_data.shape[1])
    dimension_reduction = input("Would you like to use dimension reduction? [y / n] ")
    if dimension_reduction == "y":
        dim = input("Pick a new dimension smaller than the current one ")
        PCA.reduceDim(training_data, dim)
        PCA.reduceDim(testing_data)

    boosting = input("Would you like to use boosting? [y / n] ")
    if boosting == "y":
        Boosting.CLS = classifier
        learners = input("How many learners would you like to use? [3 - 10] ")
        Boosting.boost(training_data, training_labels, testing_data, testing_labels,learners)
    else:
        classifier.train(training_data, training_labels)
        classifier.test(testing_data, testing_labels)
