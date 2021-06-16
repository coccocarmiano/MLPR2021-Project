import numpy as np
from numpy import log, pi
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from typing import Tuple


def gaussian_classifier(test_dataset: np.ndarray, means : np.ndarray, covs : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes LLRs for a binary gaussian classifier, proived the means and covariances matrices for class F and class T (in this order)
    '''
    t = np.log(1-prior_t) - np.log(prior_t)
    r, c = test_dataset.shape
    scores = np.zeros((2, c))
    cterm = r * log(2*pi)

    for i in range(2):
        _, det = np.linalg.slogdet(covs[i])
        invcov = np.linalg.inv(covs[i])
        centered = test_dataset - means[i]
        contributes = np.diag(centered.T @ invcov @ centered)
        scores[i] += -0.5 * (cterm +  det + contributes)

    llr = scores[1]-scores[0]
    return llr


def DualSVM_Train(dataset: np.ndarray, function, factr : float = 1.0, bound : float=.5):
    '''
    Trains a SVM classifier, provieded a labeled input dataset and a function.
    
    `function` can be None, in this case a linear approach is used

    `factr` is the precision used in the L-FBGS-B algorithm

    `bound` keeps alpha values between (0, `bound`)
    '''
    feats, labels = dataset[:-1, :], dataset[-1, :]
    _, c = feats.shape

    def linear(x1, x2):
        return x1.T @ x2

    if function is None:
        function = linear

    def minimize(H):
        def f(alpha):
            value = 0.5 * (alpha.T @ H @ alpha) - alpha.sum()
            gradient = (H @ alpha) - 1
            return value, gradient
        return f


    # Prepare the Z_ij matrix
    z_i = (2*labels-1)

    # Prepare the H_ij matrix
    # Exploit trace property to save time
    
    Hij = np.zeros((c, c))

    for i in range(c):
        for j in range(i+1):
            Hij[i, j] = function(dataset[:, i], dataset[:, j]) * z_i[i] * z_i[j]

    Hij = Hij -np.diag(np.diag(Hij)) + Hij.T


    zero = np.zeros(c)
    minimize_me = minimize(Hij)
    boundaries = [(0, bound) for i in range(c)]
    alphas, _, __ = fmin_l_bfgs_b(minimize_me, zero, factr=factr, maxfun=1e6, bounds=boundaries)
    return alphas

def DualSVM_Score(trdataset: np.ndarray, function, alphas : np.ndarray, tedataset: np.ndarray, bias : float=.0):
    '''
    Computes scores based on  `trdataset` support vectors and corresponding `alpha` values

    `function` can be None, in this case a linear approach is used
    '''
    def linear(x1, x2):
        return x1.T @ x2

    if function is None:
        function = linear

    trs, trl = trdataset[:-1, :], trdataset[-1, :]
    tes, tel = tedataset[:-1, :], tedataset[-1, :]

    _, trc = trs.shape
    _, tec = tes.shape

    trl = 2*trl-1
    tel = 2*tel-1

    scores = np.zeros(tec)

    for i in range(tec):
        for j in range(trc):
            scores[i] += alphas[j] * function(trs[:, j], tes[:, i]) * trl[j]

    if bias > .0:
        scores += bias
    
    return scores