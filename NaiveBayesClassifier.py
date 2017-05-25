from mnist import MNIST
import scipy.stats as stats
import numpy as np
import random

def train(training_set, labels):
    data = np.array(training_set)
    print(data.shape)
    print(len(np.mean(data, axis=0)))
    cov_mat = np.cov(data, rowvar=False)
    print(cov_mat.shape)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    W = np.matrix([eig_pairs[i][1] for i in range(16)])
    print(W.shape)
    transformed = W.dot(data.T)
    print(transformed.shape)
    cats = {i:[] for i in np.unique(labels)}
    for i, l in enumerate(labels):
        cats[l].append(transformed[:,i])
    norm_params = np.empty([3,len(cats)])
    for i, l in cats.items():
        norm_params[0,i] = np.mean(l)
        norm_params[1,i] = np.std(l)
        norm_params[2,i] = len(l)
    print(norm_params)
    
    for i in range(16):
        sample = random.randint(0,len(training_set))
        s_image = training_set[i]
        ts_image = W.dot(np.transpose(s_image))
        s_label = labels[i]
        max_label = None
        max_prob = 0
        for c in cats:
            c_mu = norm_params[0,c]
            c_sig = norm_params[1,c]
            c_siz = norm_params[2,c]
            n = stats.norm(c_mu,c_sig)
            prob = c_siz/len(labels)*math.exp(np.sum([math.log(n.pdf(x)) for x in ts_image]))
            if prob > max_prob:
                max_label = c
                max_prob = prob
        print("Label: ",s_label)
        print("Predicted: ",max_label)
            

def test(testing_set):
    pass

if __name__ == "__main__":
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    train(images, labels)

