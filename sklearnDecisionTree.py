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
        # print("Predicted")
        # print(clf.predict(features[indexToTest]))
        # print("Actual")
        # print(labels[indexToTest])
        # print()
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