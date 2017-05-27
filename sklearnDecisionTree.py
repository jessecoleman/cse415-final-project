from sklearn import tree
import numpy as np
from mnist import MNIST

def train(features, labels):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    return clf

def test(features, labels, clf):
    for indexToTest in range(10):
        print("Predicted")
        print(clf.predict(features[indexToTest]))
        print("Actual")
        print(labels[indexToTest])
        print()

if __name__ == "__main__":
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    indicesToTrain = 50
    clf = train(images[0:indicesToTrain], labels[0:indicesToTrain])
    test(images[0:indicesToTrain], labels[0:indicesToTrain], clf)