import numpy as np
from numpy import log, pi
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from typing import Tuple

def SVM_lin(dataset):
    pass

def SVM_kernel(dataset, kernel=None):
    pass

def gaussian_classifier(test_dataset: np.ndarray, means, covs, prior_t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes scores and labels from a gaussian classifiers given the covariances.
    Assumes no label feature.

    Returns matrix of scores and predictions
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
    return llr,  llr > t


def DualSVM_Train(dataset: np.ndarray, function=None, factr : float = 1.0, bound : float=.5):
    feats, labels = dataset[:-1, :], dataset[-1, :]
    r, c = feats.shape

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
    z_i = (2*labels-1).reshape((c, 1))
    Zij = z_i.T @ z_i

    # Prepare the H_ij matrix
    # Since tr(x_i^T x_j) = tr(x_j^T x_i) we know that Hij is symmetric
    # To save time we compute the lower triangular half of the matrix and then add
    # herself to the transposed to get H_ij

    Hij = np.zeros((c, c))

    for i in range(c):
        for j in range(i):
            Hij[i][j] = function(dataset[:, i], dataset[:, j])

    Hij += Hij - np.diag(np.diag(Hij)) + Hij.T
    Hij *= Zij

    zero = np.zeros(c)
    minimize_me = minimize(Hij)
    boundaries = [(0, bound) for i in range(c)]
    alphas, _, __ = fmin_l_bfgs_b(minimize_me, zero, factr=factr, bounds=boundaries)
    alphas[alphas < .0] = .0
    return alphas

def DualSVM_Score(trdataset: np.ndarray, alphas : np.ndarray, tedataset: np.ndarray, function=None, bias : float=.0):

    def linear(x1, x2):
        return x1.T @ x2

    if function is None:
        function = linear

    trs, trl = trdataset[:-1, :], trdataset[-1, :]
    tes, tel = tedataset[:-1, :], tedataset[-1, :]

    trr, trc = trs.shape
    ter, tec = tes.shape

    trl = (2*trl-1).reshape((trc, 1))
    tel = (2*tel-1).reshape((tec, 1))

    scores = np.zeros((tec, 1))

    for i in range(tec):
        for j in range(trc):
            scores[i] += alphas[i] * function(trs[:, j], tes[:, i]) * trl[i]
        scores[i] += bias
    
    return scores