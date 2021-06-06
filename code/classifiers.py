import numpy as np
from numpy import log, pi
from scipy.special import logsumexp

def SVM_lin(dataset):
    pass

def SVM_kernel(dataset, kernel=None):
    pass

def gaussian_classifier(test_dataset: np.ndarray, means, covs, prior_t: float = 0.5):
    '''
    Computes scores and labels from a gaussian classifiers given the covariances.
    Assumes centered dataset without label feature.
    '''
    scores = np.zeros((2, test_dataset.shape[1]))
    N = test_dataset.shape[0]
    cterm = N * log(2*pi)
    priors = [log(prior_t), log(1-prior_t)]

    for i in range(2):
        _, cov = np.linalg.slogdet(covs[i])
        invcov = np.linalg.inv(covs[i])
        centered = test_dataset - means[i]
        contributes = np.diag(centered.T.dot(invcov).dot(centered))
        scores[i] += -0.5 * (cterm + cov + contributes) + priors[i]#+ log(prior_t if i == 1 else 1-prior_t)

    scores = scores - logsumexp(scores, axis=0)
    return scores,  np.argmax(scores, axis=0)
