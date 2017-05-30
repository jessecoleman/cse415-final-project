# Cole Chamberlin, Ishan Saksena
# Decision Tree Classifier from sci kit learn
# CSE415, Final Project
# Spring 2017

# Validation only

from sklearn import tree
import numpy as np
from mnist import MNIST

def train(features, labels):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    return clf

def test(features, labels, clf):
    totalCorrect = 0
    for indexToTest in range(len(features)):
        resultCorrect = clf.predict(features[indexToTest]) == labels[indexToTest]
        if resultCorrect:
            totalCorrect += 1
    print(float(totalCorrect) / float(len(features)) )

if __name__ == "__main__":
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    indicesToTrain = 50000
    clf = train(images[0:indicesToTrain], labels[0:indicesToTrain])
    test(images[50000:51000], labels[50000:51000], clf)