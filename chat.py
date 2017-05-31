# Cole Chamberlin, Ishan Saksena
# Naive Bayes Classifier
# CSE415, Final Project
# Spring 2017

import NaiveBayesClassifier as nb
import DecisionTree as dt

if __name__ == "__main__":
    # Parameters for classification
    dataset_choice = input("What dataset do you want to use? [Breast Cancer (b) / Handwritten digits (h)]")
    if dataset_choice == "b":
        # training_data, training_labels, testing_data, testing_labels = get_data()
        pass
    else:
        # training_data, training_labels, testing_data, testing_labels = get_data()
        pass

    classifier_choice = input("What kind of classifier do you want to use? [Naive bayes (n) / Decision Tree (d)]")
    if classifier_choice == "n":
        #classifier =
        pass

    # TODO: Print dimensions
    print("The current dimensions of the data set are: ")
    dimension_reduction = input("Would you like to use dimension reduction? [y / n]")
    if dimension_reduction == "y":
        # reduce dimensions
        pass

    boosting = input("Would you like to use boosting? [y / n]")
    if boosting == "y":
        learners = input("How many learners would you like to use? [3 - 10]")

    # Classify input