import numpy as np
import scipy
from utils import mcol
from numpy import log, pi
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from typing import Tuple

def logreg(dataset: np.ndarray, l: float=10**-3, precision: bool=False) -> Tuple[np.ndarray, float]:
    '''
    Computes the w vector and b value for the logistic regression
    '''
    data, labels = dataset[:-1], dataset[-1]
    zi = 2*labels - 1
    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        w = w[:, None]
        #tmp = np.dot(w.T, data) + b
        #partial = (labels*np.logaddexp(0, -tmp) + (1 - labels)*np.logaddexp(0, tmp)).sum(axis=1) / data.shape[1] + l/2*np.dot(w.T, w).flatten()
        tmp = -zi*(np.dot(w.T, data) + b)
        partial = np.logaddexp(0, tmp).sum(axis=1) / data.shape[1] + l/2*np.dot(w.T, w).flatten()
        return partial
    
    max = 15000
    max_factr = 10e6
    if precision:
        max = 10e5
        max_factr = 1.0
    v0 = np.zeros(data.shape[0] + 1)
    v, _, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v0, approx_grad=True, maxfun=max, factr=max_factr)
    return v[:-1], v[-1]

def logreg_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float, t: float=0) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a Tuple with the scores and the predictions
    '''
    data, labels = evaluation_dataset[:-1], evaluation_dataset[-1]
    scores = np.dot(w.T, data) + b
    predictions = (scores > t).astype(int)
    accuracy = (predictions == labels).sum() / len(predictions)
    return (scores, predictions, accuracy)

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

    best_alphas, _, _ = scipy.optimize.fmin_l_bfgs_b(hat_Ld, alphas*C, fprime=grad_hat_Ld, args=(hat_H,), bounds=boundaries)
    w = (best_alphas*z*hat_data).sum(axis=1)
    return w[:-1], K*w[-1]

def SVM_lin_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a Tuple with the scores and the predictions
    '''
    data, labels = evaluation_dataset[:-1], evaluation_dataset[-1]
    scores = np.dot(w.T, data) + b
    predictions = (scores > 0).astype(int)
    accuracy = (predictions == labels).sum() / len(predictions)
    return (scores, predictions, accuracy)

def gaussian_ll(test_dataset: np.ndarray, mean, cov) -> np.ndarray:
    '''
    Somputes loglikelihood on a dataset given mean and covariance
    '''
    r, c = test_dataset.shape
    cterm = r * log(2*pi)

    
    _, det = np.linalg.slogdet(cov)
    try:
        invcov = np.linalg.inv(cov)
    except:
        cov = cov + 1e-10
        invcov = np.linalg.inv(cov)
    
    centered = test_dataset - mean.reshape((r, 1))
    contributes = np.diag(centered.T @ invcov @ centered)
    scores = -0.5 * (cterm +  det + contributes)

    return scores


def gaussian_classifier(test_dataset: np.ndarray, means, covs,) -> Tuple[np.ndarray, np.ndarray]:

    '''
    Computes LLRs for a binary gaussian classifier, proived the means and covariances matrices for class F and class T (in this order)
    '''
    return gaussian_ll(test_dataset, means[1], covs[1]) - gaussian_ll(test_dataset, means[0], covs[0])


def DualSVM_Train(dataset: np.ndarray, function, factr : float = 1.0, bound : float=.5):
    '''
    Trains a SVM classifier, provieded a labeled input dataset and a function.
    
    `function` can be None, in this case a linear approach is used

    `factr` is the precision used in the L-FBGS-B algorithm

    `bound` keeps alpha values between (0, `bound`)
    '''
    feats, labels = dataset[:-1, :], dataset[-1, :]
    _, c = feats.shape


    def minimize(H):
        def f(alpha):
            value = 0.5 * (alpha.T @ H @ alpha) - alpha.sum()
            gradient = (H @ alpha) - 1
            return value, gradient
        return f


    # Prepare the Z_ij matrix
    z_i = np.array(2*labels-1, dtype=int)

    # Prepare the H_ij matrix
    # Exploit trace property to save time
    
    Hij = np.zeros((c, c))

    for i in range(c):
        for j in range(i+1):
            Hij[i, j] = function(feats[:, i], feats[:, j]) * z_i[i] * z_i[j]

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


def GMM_Train(dataset, n):
    # Initialization part
    # Dummy Inizialization
    nfeatures, nsamples = dataset.shape
    dmean = dataset.mean(axis=1)
    eps = np.ones(nfeatures) * 1e-3
    means = [dmean - eps/2 + (i+1)/n * eps for i in range(n)]
    covs = [np.eye(nfeatures) for i in range(n)]
    weights = [ 1/n for i in range(n)]
        
    loglikelihoods = np.zeros((n, nsamples))

    for c in range(n):
        loglikelihoods[c] = gaussian_ll(dataset, means[c], covs[c]) + np.log(weights[c])
    lse = scipy.special.logsumexp(loglikelihoods, axis=0)


    ## Iteration part
    ll_sum_before = lse.sum()
    delta = 1
    while delta > 1e-10:
        responsabilities = np.exp(loglikelihoods-lse)
        for c in range(n):
            newmean = (responsabilities[c] * dataset).sum(axis=1) / responsabilities[c].sum()
            centered = dataset-newmean.reshape((nfeatures, 1))
            S = (responsabilities[c] * dataset) @ dataset.T / responsabilities[c].sum()
            newcov = S - newmean.reshape((nfeatures, 1)) @ newmean.reshape((nfeatures,1)).T
            newweigth = responsabilities[c].sum() / responsabilities.sum()

            means[c] = newmean
            covs[c] = newcov
            weights[c] = newweigth

        for c in range(n):
            loglikelihoods[c] = gaussian_ll(dataset, means[c], covs[c])
        lse = scipy.special.logsumexp(loglikelihoods, axis=0)

        ll_sum_after = lse.sum()
        delta = ll_sum_after-ll_sum_before
        ll_sum_before = ll_sum_after

    return weights, means, covs

def GMM_Score(dataset, weights, means, covs):
    n = len(weights)
    r, c = dataset.shape
    scores = np.zeros((n, c))

    for c in range(n):
        scores[c] = gaussian_ll(dataset, means[c], covs[c]) + np.log(weights[c])
    
    return scipy.special.logsumexp(scores, axis=0)
    