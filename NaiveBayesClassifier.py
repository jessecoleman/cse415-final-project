# Cole Chamberlin, Ishan Saksena
# Naive Bayes Classifier
# CSE415, Final Project
# Spring 2017

from mnist import MNIST
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt

class NaiveBayesClassifier:

    def __init__(self):
        self.disp_mnist = False
        self.reporting = False
        self.mean = None
        self.std = None
        self.size = None
        self.cats = None
        self.images = None

    def train(self, training_set, labels):
        global mean, std, size, cats
        # categories
        cats = {l:[] for l in np.unique(labels)}
        mean = {}
        std = {}
        size = {}
        # loop through labels to build up model
        for i, l in enumerate(labels):
            cats[l].append(training_set[i,:])
        for i, l in cats.items():
            class_data = np.array(l)
            mean[i] = np.mean(class_data,axis=0)
            std[i] = np.std(class_data,axis=0)
            size[i] = class_data.shape[0]
                  
    def test(self, testing_set, testing_labels):
        # count number of correct predictions
        correct = 0
        w = []
        r = []
        j_prob_v = np.vectorize(joint_prob)
        for i, e in enumerate(testing_set):
            s_label = testing_labels[i]
            max_label = None
            max_prob = -math.inf
            # find argmax(cats)
            for c in cats:
                c_mu = mean[c]
                c_sig = std[c]
                c_size = size[c]
                prior = c_size/len(testing_labels)
                # compute joint probability
                with np.errstate(divide='ignore'):
                    j_prob = np.nansum(j_prob_v(e, c_mu, c_sig, c_size))
                prob = math.log(prior) + j_prob
                # update prediction
                if prob > max_prob:
                    max_label = c
                    max_prob = prob
            # if prediction was correct
            r.append(max_label)
            if s_label == max_label: 
                correct += 1
                w.append(1)
            else:
                w.append(2)
            if self.reporting:
                print("Label: ",s_label)
                print("Predicted: ",max_label)
            if self.disp_mnist:
                plt.title("Label is "+str(s_label)+", Predicted "+str(max_label))
                pixels = np.array(self.images[i],dtype="uint8").reshape((28,28))
                plt.imshow(pixels,cmap='gray')
                plt.draw()
                plt.pause(0.01)
                plt.cla()

        print("Accuracy:",correct/len(testing_labels))
        print("Total correct:",correct)
        print("Total tested:",len(testing_labels))
        return r, w

def joint_prob(x,mu,sigma,size):
    if sigma==0 and mu==x: return 0.0
    try:
        return math.log(stats.norm(mu,sigma).pdf(x))
    except:
        return math.log(1/size)
