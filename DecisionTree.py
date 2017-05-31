# Cole Chamberlin, Ishan Saksena
# Decision Tree Classifier
# CSE415, Final Project
# Spring 2017

import numpy as np

class DecisionTree:

    def __init__(self):
        self.classifier = {}

    # Entropy above current node
    def entropy(self, s):
        res = 0
        val, counts = np.unique(s, return_counts=True)
        freqs = counts.astype('float') / len(s)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res

    # Dictionary from element to index, unique
    def partition(self, a):
        return {c: (a==c).nonzero()[0] for c in np.unique(a)}


    def information_gain(self, y, x):

        # Entropy so far
        res = self.entropy(y)

        # Splitting on attribute x
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float') / len(x)

        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            try:
                res -= p * self.entropy(y[x == v])
            except:
                print(x == v)

        return res

    # Returns true if the array contains only one unique element
    def is_pure(self, s):
        return len(set(s)) == 1

    # Split remaining cases based on information gain
    def train(self, x, y):
        # Return if pure or empty
        if self.is_pure(y) or len(y) == 0:
            return y

        # Select attribute that gives highest information gain
        gain = np.array([self.information_gain(y, x_attr) for x_attr in x.T])
        selected_attr = np.argmax(gain)

        # Return selection if no gain
        if np.all(gain < 1e-6):
        # if np.all(gain < 0.5):
            return y

        # Split using the selected attribute
        sets = self.partition(x[:, selected_attr])

        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)

            subclassifier = self.train(x_subset, y_subset)

            if selected_attr not in self.classifier:
                self.classifier[selected_attr] = {k: subclassifier}
            else:
                self.classifier[selected_attr][k] = subclassifier

    # Traverse classifier tree and return class
    def classify(self, classifier, testcase):
        # Return class if it's an array
        if isinstance(classifier, (list, np.ndarray)) and self.is_pure(classifier):
            return classifier[0]

        # Always single key, which attribute to choose next
        for key in classifier.keys():
            values = classifier[key]

            # Find the closest previously seen value
            value = min(values, key=lambda x: abs(x-testcase[key]))
            return self.classify(classifier[key][value], testcase)

    # Prints the accuracy as % of classifier
    def test(self, features, labels):
        totalCorrect = 0
        r = []
        w = []
        for indexToTest in range(len(features)):
            result = self.classify(self.classifier, features[indexToTest])
            r.append(result)
            resultCorrect = result == labels[indexToTest]
            if resultCorrect:
                totalCorrect += 1
                w.append(1)
            else:
                w.append(1.5)

        # Report Results
        print("Percent Correct:",totalCorrect/(len(features))
        print("Total Correct:",totalCorrect)
        print("Total Tested ",len(features))
        return r, w

# Run the classifier with different data sets
if __name__ == "__main__":
    pass
