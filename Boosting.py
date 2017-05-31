# Cole Chamberlin, Ishan Saksena
# Boosting meta-algorithm
# CSE415, Final Project
# Spring 2017

import random
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt

CLS = dt

def boost(training_set, training_labels, testing_set, testing_labels, num_learners):
    dt.reporting = False
    nbc.reporting = False
    ensemble = []
    w = [1] * len(training_set)
    idx = list(range(len(training_set)))
    print(idx)
    for i in range(num_learners):
        # alternate classifiers
        if i % 2 == 0:
            ensemble.append(dt())
        else:
            ensemble.append(nbc())
        # take sample for current learner
        s = random.choices(idx, weights=w, k=int(len(training_set)*0.6))
        sample_set = np.array([training_set[j] for j in s])
        sample_labels = np.array([training_labels[j] for j in s])
        # train learner with sample
        ensemble[i].train(sample_set, sample_labels)
        # test learner with entire training set
        r, w = ensemble[i].test(training_set, training_labels)

    results = []
    # test on testing set with all learners
    for i,n in enumerate(ensemble):
        results.append(n.test(testing_set, testing_labels)[0])
    results = np.array(results).T
    correct = 0
    # count votes
    for r in range(results.shape[0]):
        u, indices = np.unique(results, return_inverse=True)
        pred = u[np.argmax(np.bincount(indices))]
        print("Label:",testing_labels[r])
        print("Predicted:",pred)
        if testing_labels[r] == pred: correct += 1
    
    # display results
    print("Accuracy:",correct/(len(testing_set)))
    print("Total correct:",correct)
    print("Total tested ",len(testing_set))

if __name__ == '__main__':
    import BreastCancer as bc
    training_set, training_labels, testing_set, testing_labels = bc.get_data()
    d = dt()
    d.train(training_set, training_labels)
    d.test(testing_set, testing_labels)
    boost(training_set, training_labels, testing_set, testing_labels, 1)
