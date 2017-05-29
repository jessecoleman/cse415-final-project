# Ishan Saksena
# Decision Tree Classifier
# CSE415, Final Project
# Spring 2017

import csv
from mnist import MNIST

import numpy as np
from pprint import pprint

x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y, {}

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y, {}


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    classifier = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        # Comment out string part, keep only tuple to traverse
        #
        split, subclassifier = recursive_split(x_subset, y_subset)
        res["x_%d = %d" % (selected_attr, k)] = split
        if selected_attr not in classifier:
            classifier[selected_attr] = {k: split}
        else:
            if subclassifier:
                classifier[selected_attr][k] = subclassifier
            else:
                classifier[selected_attr][k] = split
    return res, classifier

def classify(features, classifier, testcase):
    # Runs on case to get which value to look at
    for key in classifier:
        value = testcase[key]
        # Check if it's an array
        if isinstance(classifier, (list, tuple, np.ndarray)) and is_pure(classifier):
            return classifier[0]
        if classifier:
            return classify(features, classifier[key][testcase[key]], testcase)

X = np.array([x1, x2]).T
res, classifier = recursive_split(X, y)
print("res is ")
pprint(res)
print()
# print("The classifier is ")
# pprint(classifier)
for i in range(len(X)):
    print(classify(X, classifier, X[i]) == y[i])

if __name__ == "__main__":
    # Load 2016-us-election data
    # Source: https://www.kaggle.com/benhamner/2016-us-election
    # with open('2016-us-election/county_facts.csv', 'r') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in spamreader:
    #         print(', '.join(row))

    # Load MNIST data set
    # Images of size 28, 28
    # mndata = MNIST('samples')
    # images, labels = mndata.load_training()
    # images_training = np.array(images[1:800])
    # labels_training = np.array(labels[1:800])
    # pprint(recursive_split(images_training, labels_training))
    # print(labels_training[0])
    pass