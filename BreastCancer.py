import pandas as pd
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt

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


data = pd.read_csv("data.csv")
labels = np.array(data.ix[1:,1])
data = np.array(data.ix[1:,2:32])

mask = np.ones(data.shape[0], dtype=bool)
for i in range(int(data.shape[0]*0.25)):
    mask[i] = 0

np.random.shuffle(mask)
training_data = data[mask]
training_labels = labels[mask]
testing_data = data[np.invert(mask)]
testing_labels = labels[np.invert(mask)]

c1 = nbc()
reduceDim(training_data, 12)
reduceDim(testing_data)
c1.train(training_data, training_labels)
c1.test(testing_data, testing_labels)
c2 = dt()
c2.train(training_data, training_labels)
c2.test(testing_data, testing_labels)


