from mnist import MNIST
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt

MNIST = False
REPORTING = True
W = None

class NaiveBayesClassifier:
    def __init__(self):
        self.mean = None
        self.std = None
        self.size = None
        self.cats = None
        self.images = None

    def train(self, training_set, labels):
        global mean, std, size, cats
        # categories
        cats = {i:[] for i in np.unique(labels)}
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
        w = [1] * len(testing_set)
        r = [None] * len(testing_set)
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
            r[i] = max_label
            if s_label == max_label: 
                correct += 1
            else:
                w[i] = 1.5
            if MNIST:
                plt.title("Label is "+s_label+", Predicted "+max_label)
                pixels = np.array(images[i],dtype="uint8").reshape((28,28))
                plt.imshow(pixels,cmap='gray')
                plt.draw()
                plt.pause(0.01)
            if REPORTING:
                print("Label: ",s_label)
                print("Predicted: ",max_label)
            
        print("accuracy: ",correct/len(testing_labels))
        return r, w

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
            testing_set = W.dot(testing_set.T).T.real
        train(training_set, training_labels)
        test(testing_set, testing_labels)

def reduceDim(data, small):
    global W
    cov_mat = np.cov(data, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    W = np.matrix([eig_pairs[i][1] for i in range(small)])
    transformed = W.dot(data.T).T
    return transformed.real

def joint_prob(x,mu,sigma,size):
    if sigma==0 and mu==x: return 0.0
    try:
        return math.log(stats.norm(mu,sigma).pdf(x))
    except:
        return math.log(1/(size*1000000000000))

if __name__ == '__main__':
    MNIST = True
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    k_fold(np.array(images), np.array(labels), 100, 28)
