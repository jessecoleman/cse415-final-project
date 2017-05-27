from mnist import MNIST
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time

W = None
mean = None
std = None
size = None
cats = None
images = None

REPORTING = True

def train(training_set, labels):
    global mean, std, size, cats
    # categories
    cats = {i:[] for i in np.unique(labels)}
    mean = np.empty([len(cats),training_set.shape[1]])
    std = np.empty([len(cats),training_set.shape[1]])
    size = np.empty([len(cats)])
    # loop through labels to build up model
    for i, l in enumerate(labels):
        cats[l].append(training_set[i,:])
    for i, l in cats.items():
        class_data = np.array(l)
        mean[i] = np.mean(class_data.real,axis=0)
        std[i] = np.std(class_data.real,axis=0)
        size[i] = class_data.shape[0]
              
def joint_prob(x,mu,sigma,size):
    if sigma==0 and mu==x: return 0.0
    try:
        return math.log(stats.norm(mu,sigma).pdf(x))
    except:
        return math.log(1/(size*1000000000000))

def test(testing_set, testing_labels):
    # count number of correct predictions
    correct = 0
    j_prob_v = np.vectorize(joint_prob)
    for i, s in enumerate(testing_set):
        s_image = s
        s_label = testing_labels[i]
        max_label = None
        max_prob = -math.inf
        for c in cats:
            c_mu = mean[c]
            c_sig = std[c]
            c_size = size[c]
            prior = c_size/len(labels)
            # compute joint probability
            with np.errstate(divide='ignore'):
                j_prob = np.nansum(j_prob_v(s_image, c_mu, c_sig, c_size))
            prob = math.log(prior) + j_prob
            # update prediction
            if prob > max_prob:
                max_label = c
                max_prob = prob
        if REPORTING:
            plt.title("Label is {label}, Predicted {pred}".format(label=s_label,pred=max_label))
            pixels = np.array(images[i],dtype="uint8").reshape((28,28))
            plt.imshow(pixels,cmap='gray')
            plt.draw()
            plt.pause(0.01)
            print("Label: ",s_label)
            print("Predicted: ",max_label)
        if s_label == max_label: correct += 1
    print("accuracy: ",correct/len(testing_labels))

def k_fold(data, labels, k_folds):
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
        train(training_set, training_labels)
        test(testing_set, testing_labels)

def reduceDim(data, small):
    cov_mat = np.cov(data, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    global W
    W = np.matrix([eig_pairs[i][1] for i in range(small)])
    transformed = W.dot(data.T)
    return transformed.T.real
 
if __name__ == "__main__":
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    transformed = reduceDim(np.array(images), 28)
    k_fold(transformed, np.array(labels), 100)
    test(transformed[0:100])
