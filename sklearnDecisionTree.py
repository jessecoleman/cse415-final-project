# Cole Chamberlin, Ishan Saksena
# Decision Tree Classifier from sci kit learn
# CSE415, Final Project
# Spring 2017

# Validation only

from sklearn import tree
import numpy as np
from mnist import MNIST

def train(features, labels):
    clf = tree.DecisionTreeClassifier("gini")
    clf.fit(features, labels)
    return clf

# Prints the accuracy as % of classifier
def test(features, labels, clf):
    totalCorrect = 0
    for indexToTest in range(len(features)):
        resultCorrect = clf.predict(features[indexToTest]) == labels[indexToTest]
        if resultCorrect:
            totalCorrect += 1

    # Report Results
    print("Percent Correct: ", end = " ")
    print(float(totalCorrect) / float(len(features)) )
    print("Total Correct: ", end=" ")
    print(totalCorrect)
    print("Total Tested ", end=" ")
    print(len(features))

if __name__ == "__main__":
    mndata = MNIST('samples')
    images, labels = mndata.load_training()


    #test(images[50000:51000], labels[50000:51000], clf)

    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    print(images[0:40])
    # images = reduceDim(images, 28)

    indicesToTrain = 5000
    indicesToTestUntil = 6000
    images_training = images[0:indicesToTrain]
    labels_training = labels[0:indicesToTrain]

    images_testing = images[indicesToTrain:indicesToTestUntil]
    labels_testing = labels[indicesToTrain:indicesToTestUntil]

    clf = train(images[0:indicesToTrain], labels[0:indicesToTrain])
    #classifier = recursive_split(images_training, labels_training)
    # pprint(classifier)
    # print(classify(classifier, images_training[1]) == labels_training[1])
    test(images_testing, labels_testing, clf)