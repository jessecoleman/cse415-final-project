from mnist import MNIST
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt
import random
#import DecisionTree as c2

CLS = dt
W = None

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
        print(w)

    #red_testing_set = reduceDim(testing_set)
    CLS.REPORTING = True
    results = []
    for i,n in enumerate(ensemble):
        results.append(n.test(testing_set, testing_labels)[0])
    results = np.array(results).T
    print(results.shape)
    correct = 0
    for r in range(results.shape[0]):
        pred = np.bincount(results[r]).argmax()
        print("label:",testing_labels[r])
        print("pred:",pred)
        if testing_labels[i] == pred: correct += 1
    
    print(correct)
    print(len(testing_set))
    print("accuracy:",correct/len(testing_set))

def reduceDim(data, small=None):
    global W
    if small != None:
        cov_mat = np.cov(data, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        W = np.matrix([eig_pairs[i][1] for i in range(small)])
    return W.dot(data.T).T.real

if __name__ == '__main__':
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    boost(np.array(images[:400]), np.array(images[400:500]), np.array(labels[:400]), \
            np.array(labels[400:500]), 3)
