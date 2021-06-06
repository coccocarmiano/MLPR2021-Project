import numpy as np
from numpy import log, pi
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b as minimize

def SVM_lin(dataset):
    pass

def SVM_kernel(dataset, kernel=None):
    pass

def gaussian_classifier(test_dataset: np.ndarray, means, covs, prior_t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    '''
    Computes scores and labels from a gaussian classifiers given the covariances.
    Assumes no label feature.

    Returns matrix of scores and predictions
    '''
    scores = np.zeros((len(means), test_dataset.shape[1]))
    N = test_dataset.shape[0]
    cterm = N * log(2*pi)
    priors = [log(prior_t), log(1-prior_t)]

    for i in range(len(means)):
        _, cov = np.linalg.slogdet(covs[i])
        invcov = np.linalg.inv(covs[i])
        centered = test_dataset - means[i]
        contributes = np.diag(centered.T.dot(invcov).dot(centered))
        scores[i] += -0.5 * (cterm + cov + contributes) + priors[i]

    scores = scores - logsumexp(scores, axis=0)
    return scores,  np.argmax(scores, axis=0)


def RBF_SVM(dataset : np.ndarray, test_dataset: np.ndarray, gamma : float=1., reg_bias : float=0., boundary : float=1.) -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Returns a tuple containing:
        -> 0: scores of the trained SVM
        -> 1: predictions of the trained SVM
        -> 2: accuracy (not percentage) of the trained SVM

    `dataset` and `test_dataset` must include feature label (last one).

    `gamma` is the gamma RBF parameter, use intermediate values (1.0, ..., 10.0)

    `reg_bias` is the regularized bias term. Use small values (0.0, ..., 1.0)

    `boundary` for `alpha` terms, constraining to `0 - boundary`. Use small values (0.0, ..., 1.0)
    '''
    # Preparing our Hij matrix
    features, labels = dataset[:-1, :], dataset[-1, :]
    r, c = features.shape
    zlabels = (2*labels-1).reshape((1, c))
    Zij = zlabels.T.dot(zlabels)
    kernmat = np.zeros((c, c))
    
    # Basically we exploit broadcasting to compute `c` times a matrix
    # Which contains the diff of sample `i` with all other
    # Then we sum along the rows to compute the `i-th` row of matrix H
    for idx in range(c):
        xi = features[:, idx].reshape((r, 1))
        diff = features - xi
        norm = (diff * diff).sum(axis=0)
        kernmat[idx] += norm
    
    kernmat *= -gamma
    kernmat = np.exp(kernmat)
    kernmat += reg_bias
    Hij = Zij * kernmat
    
    
    # Here do the actual minimization (maximization)
    def to_minimize(alpha):
        value = 0.5 * alpha.T.dot(Hij).dot(alpha) - alpha.sum()
        gradient = Hij.dot(alpha)-1
        return (value, gradient)
    
    boundaries = [(0, boundary) for elem in range(c)]
    start = np.zeros(c)
    alphas, _, __ = minimize(to_minimize, start, bounds=boundaries)
    alphas = np.array(alphas > 0., dtype=float)

    # Here we begin the scoring part
    test_dataset, test_labels = test_dataset[:-1, :], test_dataset[-1, :]
    r, c = test_dataset.shape
    scores = np.zeros(c)

    # Here we do a process very similar to the one done before
    for idx in range(c):
        x_t = test_dataset[:, idx].reshape((r, 1))
        diff = features - x_t
        norm = (diff * diff).sum(axis=0)
        expterm = norm * -gamma
        expterm = np.exp(expterm)
        expterm = expterm + reg_bias
        summatory = expterm * alphas * zlabels
        scores[idx] = summatory.sum()

    predictions = scores > 0
    accuracy = (predictions == test_labels).sum() / len(predictions)

    return scores, scores > 0, accuracy
