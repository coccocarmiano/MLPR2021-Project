import numpy as np
import scipy
from utils import mcol
from numpy import log, pi
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from typing import Tuple

def logreg(dataset: np.ndarray, l: float=10e-3) -> Tuple[np.ndarray, float]:
    '''
    Computes the w vector and b value for the logistic regression
    '''
    data, labels = dataset[:-1], dataset[-1]

    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        w = mcol(w)
        # computes objective function
        partial = 0
        for i in range(data.shape[1]):
            partial = partial + labels[i]*np.log1p(np.exp(- np.dot(w.T, data[:, i]) - b)) + (1 - labels[i])*np.log1p(np.exp(np.dot(w.T, data[:, i]) + b))
        partial =  partial/data.shape[1] + l/2*np.dot(w.T, w).flatten()

        return partial

    v0 = np.zeros(data.shape[0] + 1)
    v, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v0, approx_grad=True, factr=0)
    return v[:-1], v[-1]

def logreg_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a tuple with the scores and the predictions
    '''
    data = evaluation_dataset[:-1]
    scores = np.dot(w.T, data) + b
    predictions = (scores > 0).astype(int)  
    return (scores, predictions)

def SVM_lin(dataset: np.ndarray, K: float, C: float) -> Tuple[np.ndarray, float]:
    '''
    Computes the w vector and b value for linear SVM 
    '''
    data, labels = dataset[:-1], dataset[-1]

    z = (2*labels - 1).reshape(1, labels.shape[0]) # zi = +-1, 1 if belongs to class Ht, -1 if belongs to class Hf
    
    hat_data = np.vstack([data, K*np.ones(data.shape[1])])

    hat_H = np.dot(z.T*hat_data.T, z*hat_data)

    alphas = np.ones((hat_data.shape[1], 1))/2

    boundaries = [(.0, C) for _ in alphas]

    def grad_hat_Ld(alphas, hat_H):
        return (np.dot(hat_H, alphas) - 1)

    def hat_Ld(alphas, hat_H):
        return  -(-1/2*np.dot(alphas.T, np.dot(hat_H, alphas)) + np.dot(alphas.T, np.ones(alphas.shape[0]))).item()

    best_alphas, _, _ = scipy.optimize.fmin_l_bfgs_b(hat_Ld, alphas*C, fprime=grad_hat_Ld, args=(hat_H,), bounds=boundaries, factr=0.0)
    w = (best_alphas*z*hat_data).sum(axis=1)
    return w[:-1], K*w[-1]

def SVM_lin_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a tuple with the scores and the predictions
    '''
    data = evaluation_dataset[:-1]
    scores = np.dot(w.T, data) + b
    predictions = (scores > 0).astype(int)
    return (scores, predictions)

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