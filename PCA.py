# Cole Chamberlin, Ishan Saksena
# Principal Component Analysis
# CSE415, Final Project
# Spring 2017

import numpy as np

W = None

def reduceDim(data, small=None):
    global W
    if small != None:
        cov_mat = np.cov(data, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        W = np.matrix([eig_pairs[i][1] for i in range(small)])
    return W.dot(data.T).T.real

def trainTestSplit(data, labels, split):
    mask = np.ones(data.shape[0], dtype=bool)
    for i in range(int(data.shape[0]*(1-split))):
        mask[i] = 0
    np.random.shuffle(mask)
    training_data = data[mask]
    training_labels = labels[mask]
    testing_data = data[np.invert(mask)]
    testing_labels = labels[np.invert(mask)]
    return training_data, training_labels, testing_data, testing_labels

def k_fold(data, labels, k_folds, dim_red=False):
    fold_len = data.shape[0] // k_folds
    for k in range(k_folds):
        mask = np.ones(len(data),np.bool)
        mask[k*fold_len:(k+1)*fold_len] = 0
        # training split
        training_set = data[mask]
        training_labels = labels[mask]
        # testing split
        testing_set = data[np.invert(mask)]
        testing_labels = labels[np.invert(mask)]
        # reduce dimension
        if dim_red > 0:
            training_set = reduceDim(training_set, dim_red)
            testing_set = reduceDim(testing_set)
        train(training_set, training_labels)
        test(testing_set, testing_labels)




