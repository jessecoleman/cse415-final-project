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
    while(True):
        # Parameters for classification
        dataset_choice = input("What dataset do you want to use? [Breast Cancer (b) / Handwritten digits (h)] ")
        if dataset_choice == "b":
            training_data, training_labels, testing_data, testing_labels = BreastCancer.get_data()
        else:
            training_data, training_labels, testing_data, testing_labels = mnist_data.get_data()

        classifier_choice = input("What kind of classifier do you want to use? [Naive bayes (n) / Decision Tree (d)] ")
        if classifier_choice == "n":
            classifier = nb()
            print("The current dimension of the dataset is:", training_data.shape[1])
            dimension_reduction = input("Do you want to reduce the dimension with PCA? [y / n] ")
            if dimension_reduction == "y":
                dim = int(input("Pick a new dimension smaller than the current one "))
                training_data = PCA.reduceDim(training_data, dim)
                testing_data = PCA.reduceDim(testing_data)
        else:
            classifier = dt()
        
        disp = input("Would you like to see the results as they're classified? [y / n] ")
        if disp == "y":
            classifier.reporting = True
        
        boosting = input("Would you like to use boosting? [y / n] ")
        if boosting == "y":
            Boosting.CLS = classifier
            learners = input("How many learners would you like to use? [3 - 10] ")
            Boosting.boost(training_data, training_labels, testing_data, testing_labels,learners)
        else:
            classifier.train(training_data, training_labels)
            classifier.test(testing_data, testing_labels)
