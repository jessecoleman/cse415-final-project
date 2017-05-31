# Cole Chamberlin, Ishan Saksena
# Boosting meta-algorithm
# CSE415, Final Project
# Spring 2017

import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt
import PCA
import random
#import DecisionTree as c2

CLS = dt

def boost(training_set, testing_set, training_labels, testing_labels, num_learners):
    #CLS.REPORTING = False
    ensemble = []
    w = [1] * len(training_set)
    idx = list(range(len(training_set)))
    for i in range(num_learners):
        ensemble.append(CLS())
        # sample for each bag
        s = random.choices(idx, weights=w, k=int(len(training_set)*0.6))
        sample_set = np.array([training_set[j] for j in s])
        sample_labels = np.array([training_labels[j] for j in s])
        #sample_set = reduceDim(sample_set, 28)
        #red_training_set = reduceDim(training_set)
        ensemble[i].train(sample_set, sample_labels)
        r, w = ensemble[i].test(training_set, training_labels)

    #red_testing_set = reduceDim(testing_set)
    #CLS.REPORTING = True
    results = []
    for i,n in enumerate(ensemble):
        results.append(n.test(testing_set, testing_labels)[0])
    results = np.array(results).T
    correct = 0
    for r in range(results.shape[0]):
        u, indices = np.unique(results, return_inverse=True)
        pred = u[np.argmax(np.bincount(indices))]
        print("label:",testing_labels[r])
        print("pred:",pred)
        if testing_labels[r] == pred: correct += 1
    
    print("accuracy:",correct/len(testing_set))

if __name__ == '__main__':
    pass 
    #mndata = MNIST('samples')
    #images, labels = mndata.load_training()
    #boost(np.array(images[:400]), np.array(images[400:500]), np.array(labels[:400]), \
    #        np.array(labels[400:500]), 3)
