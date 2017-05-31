import pandas as pd
import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier as nbc
from DecisionTree import DecisionTree as dt
data = pd.read_csv("data.csv")

print(data.shape)
# print(data.head())
# print(data.columns)

labels = np.array(data.ix[1:,1])
data = np.array(data.ix[1:,2:31])

mask = np.ones(data.shape[0], dtype=bool)
for i in range(int(data.shape[0]*0.25)):
    mask[i] = 0

np.random.shuffle(mask)
training_data = data[mask]
training_labels = labels[mask]
testing_data = data[np.invert(mask)]
testing_labels = labels[np.invert(mask)]

#print(testing_data.shape)
# print(training_data.shape)
# print(training_labels.shape)

# c1 = nbc()
# c1.train(training_data, testing_labels)
# c1.test(testing_data, testing_labels)
#
c2 = dt()
c2.train(training_data, training_labels)
c2.test(testing_data, testing_labels)

