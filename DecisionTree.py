# Cole Chamberlin, Ishan Saksena
# Decision Tree Classifier
# CSE415, Final Project
# Spring 2017

import csv
from mnist import MNIST

import numpy as np
from pprint import pprint

class DecisionTree:
    # Entropy above current node
    def entropy(self, s):
        res = 0
        val, counts = np.unique(s, return_counts=True)
        freqs = counts.astype('float') / len(s)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res

    # Dictionary from element to index, unique
    def partition(self, a):
        return {c: (a==c).nonzero()[0] for c in np.unique(a)}


    def information_gain(self, y, x):

        # Entropy so far
        res = self.entropy(y)

        # Splitting on attribute x
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float') / len(x)

        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])

        return res

    # Returns true if the array contains only one unique element
    def is_pure(self, s):
        return len(set(s)) == 1

    # Split remaining cases based on information gain
    def recursive_split(self, x, y):
        # Return if pure or empty
        if self.is_pure(y) or len(y) == 0:
            return y

        # Select attribute that gives highest information gain
        gain = np.array([self.information_gain(y, x_attr) for x_attr in x.T])
        selected_attr = np.argmax(gain)

        # Return selection if no gain
        if np.all(gain < 1e-6):
        # if np.all(gain < 0.5):
            return y

        # Split using the selected attribute
        sets = self.partition(x[:, selected_attr])

        classifier = {}
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)

            subclassifier = self.recursive_split(x_subset, y_subset)

            if selected_attr not in classifier:
                classifier[selected_attr] = {k: subclassifier}
            else:
                classifier[selected_attr][k] = subclassifier
        return classifier

    # Traverse classifier tree and return class
    def classify(self, classifier, testcase):
        # Return class if it's an array
        if isinstance(classifier, (list, np.ndarray)) and self.is_pure(classifier):
            return classifier[0]

        # Always single key, which attribute to choose next
        for key in classifier.keys():
            values = classifier[key]

            # Find the closest previously seen value
            value = min(values, key=lambda x: abs(x-testcase[key]))
            return self.classify(classifier[key][value], testcase)

    # Prints the accuracy as % of classifier
    def test(self, features, labels, clf):
        totalCorrect = 0
        for indexToTest in range(len(features)):
            resultCorrect = self.classify(clf, features[indexToTest]) == labels[indexToTest]
            if resultCorrect:
                totalCorrect += 1

        # Report Results
        print("Percent Correct: ", end=" ")
        print(float(totalCorrect) / float(len(features)))
        print("Total Correct: ", end=" ")
        print(totalCorrect)
        print("Total Tested ", end=" ")
        print(len(features))

# PCA
def reduceDim(data, small):
    global W
    cov_mat = np.cov(data, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    W = np.matrix([eig_pairs[i][1] for i in range(small)])
    transformed = W.dot(data.T).T
    return transformed.real


# Run the classifier with different data sets
if __name__ == "__main__":
    dt = DecisionTree()

    # #
    # Load 2016-us-election data
    # Source: https://www.kaggle.com/benhamner/2016-us-election
    # #
    # with open('2016-us-election/county_facts.csv', 'r') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in spamreader:
    #         print(', '.join(row))

    #
    # Load MNIST data set
    # Images of size 28, 28
    #
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    images = reduceDim(images, 28)

    indicesToTrain = 500
    indicesToTestUntil = 600
    images_training = images[0:indicesToTrain]
    labels_training = labels[0:indicesToTrain]

    images_testing = images[indicesToTrain:indicesToTestUntil]
    labels_testing = labels[indicesToTrain:indicesToTestUntil]

    classifier = dt.recursive_split(images_training, labels_training)
    #pprint(classifier)
    #print(classify(classifier, images_training[1]) == labels_training[1])
    dt.test(images_testing, labels_testing, classifier)


    # #
    # # Original example
    # #
    # x1 = [0, 1, 1, 2, 2, 2]
    # x2 = [0, 0, 1, 1, 1, 0]
    # x3 = [0, 0, 3, 4, 5, 0]
    # y = np.array([0, 0, 0, 1, 1, 0])
    #
    # X = np.array([x1, x2, x3]).T
    # classifier = dt.recursive_split(X, y)
    # print()
    # print("The classifier is ")
    # pprint(classifier)
    # for i in range(len(X)):
    #      pprint(dt.classify(classifier, X[i]) == y[i])

    pass