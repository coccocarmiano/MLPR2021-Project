import numpy as np
import scipy as sp
import matplotlib.patches as ptc
from typing import List, Tuple
from scipy import stats

ppf = stats.norm.ppf

train_data_file = '../data/Train.txt'
test_data_file = '../data/Test.txt'
nfeats = 11

green = '#00b330'
"""
The color green choosen for plotting
"""

red = '#b30000'
"""
The color red choosen for plotting
"""


def load_train_data():
    '''
    Returns the data from `Train.txt` organized as column samples. Last field is label.
    '''
    train_file = open(train_data_file, 'r')
    lines = [line.strip() for line in train_file]
    train_file.close()

    splits = []
    for line in lines:
        split = line.split(',')
        arr = np.array([float(i) for i in split])
        splits.append(arr)
    matrix = np.array(splits).T
    return matrix


def load_test_data():
    '''
    Returns the data from `Test.txt` organized as column samples. Last field is label.
    '''
    test_fil = open(test_data_file, 'r')
    lines = [line.strip() for line in test_fil]
    test_fil.close()

    splits = []
    for line in lines:
        split = line.split(',')
        arr = np.array([float(i) for i in split])
        splits.append(arr)
    matrix = np.array(splits).T
    return matrix


def PCA(dataset: np.ndarray, feat_label: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Execute eigenvalue decompsition on `dataset`.

    Parameters:

    `dataset`: numpy matrix of column samples. Assumes last feat to label.

    `feat_label`: if false assumes no label feature

    Returns:

    The (already sorted) array `w` of eigenvalues and `v` of eigenvectors.
    '''

    if feat_label:
        feats = dataset[:-1, :]
    else:
        feats = dataset

    cov = fc_cov(feats)
    w, v = np.linalg.eigh(cov)
    w, v = w[::-1], v[:, ::-1]

    return w, v

def reduce_dataset(dataset, n=None):
    samples, labels = dataset[:-1], dataset[-1]
    r, c = samples.shape
    if n == None:
        n = r
    elif n == r:
        return dataset
    
    _, v = PCA(samples, feat_label=False)
    vt = v[:, :n]
    reduced_samples = vt.T @ samples
    reduced_dataset = np.vstack([reduced_samples, labels])
    return reduced_dataset

def get_patches() -> List[ptc.Patch]:
    '''
    Returns: patches list to be passed to either `legend` or `figlegend` (to the `handles` attribute).
    '''
    bpatch = ptc.Patch(color=red, label="Bad Wine")
    gpatch = ptc.Patch(color=green, label="Good Wine")
    return [gpatch, bpatch]


def kfold(dataset: np.ndarray, n: int = 5) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    '''
    Splits a dataset. Returns a tuple.

    First element is `n` evenly split subsets of `dataset`.

    Second element are `n` tuples. Each tuple contains in the first element a subset of
    `N-1` parts of `dataset` and in the second the remaining one.
    '''
    if (len(dataset.shape) > 2):
        print("Error: Wrong Dataset Shape")
        exit()

    _, c = dataset.shape
    frac = c / n
    splits = []

    for i in range(n):
        a = int(i*frac)
        b = int((i+1)*frac)
        fold = dataset[:, a:b]
        splits.append(fold)

    e = np.empty(n, dtype=object)
    for i in range(n):
        e[i] = splits[i]
    splits = e

    folds = []
    sel = np.arange(n)
    for i in range(n):
        train, test = splits[sel != i], splits[sel == i]
        train = np.concatenate(train, axis=1)
        test = np.concatenate(test, axis=1)
        folds.append((train, test))
    splits = [split for split in splits]
    return splits, folds

def mcol(v):
    '''
    Reshape a 1D array to a column vector with same dimension
    '''
    return v.reshape((v.shape[0], 1))


def fc_mean(dataset: np.ndarray) -> np.ndarray:
    '''
    Returns the mean of a dataset along the columns. 
    `dataset` has to be of one class only.
    Assumes no label feat.
    '''
    r, _ = dataset.shape
    return dataset.mean(axis=1).reshape((r, 1))


def fc_cov(dataset: np.ndarray) -> np.ndarray:
    '''
    Returns the covariance matrix of a dataset.
    `dataset` has to be of one class only
    Assumes no label feat.
    '''
    _, c = dataset.shape
    mean = fc_mean(dataset)
    cent = dataset - mean
    cov = cent @ cent.T / c
    return cov

def fc_std(dataset: np.ndarray) -> np.ndarray:
    '''
    '''
    r, _ = dataset.shape
    std = dataset.std(axis=1).reshape((r, 1))
    return std

def DCF(predictions: np.ndarray, labels: np.ndarray, prior_t: float = 0.5, costs: Tuple[float, float] = (1., 1.)) -> float:
    '''
    Returns the normalized and unnormalized DCF values

    `predictions` are the predicted samples

    `labels` are the actual samples labels

    `prior_t` is P(C = 1) (default: 0.5)

    `costs` is a tuple containing (P(C = 0), P(C = 1)) (default: (1, 1))
    '''
    FP = ((predictions == 1) == (labels == 0)).sum()
    FN = ((predictions == 0) == (labels == 1)).sum()
    TP = ((predictions == 1) == (labels == 1)).sum()
    TN = ((predictions == 0) == (labels == 0)).sum()

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    unnorm_dcf = FNR*costs[0]*prior_t + FPR * costs[1] * (1-prior_t)
    norm_dcf = unnorm_dcf / min(prior_t * costs[0], (1-prior_t) * costs[1])

    return (norm_dcf, unnorm_dcf)


def minDCF(scores : np.ndarray, labels : np.ndarray, prior_t : float=.5, thresholds : np.ndarray = None) -> Tuple[float, float, np.ndarray]:
    '''
    Computes minDCF and optimal threshold for the given prior

    If `thresholds` is None defaults to 1000 default thresholds
    '''

    mindcf = np.Inf
    best_threshold = .0

    if thresholds is None:
        thresholds = np.sort(scores)

    for threshold in thresholds:
        pred = scores > threshold
        dcf, _ = DCF(pred, labels, prior_t=prior_t)

        if dcf < mindcf:
            mindcf = dcf
            best_threshold = threshold
    
    return mindcf, best_threshold


def normalize(dataset: np.ndarray, other: np.ndarray = None) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
    '''
    Z-Normalize a (two) LABELED dataset(s).

    If `other` is provided, normalizes it with the data from `dataset` and returns a tuple with normalized
    `dataset, other`, otherwise just normalized dataset.
    '''
    
    dataset, dataset_labels = dataset[:-1, :], dataset[-1, :]
    if other is not None:
        other, other_labels = other[:-1, :], other[-1, :]

    r, _ = dataset.shape
    mean = fc_mean(dataset)
    std = dataset.std(axis=1).reshape((r, 1))
    dataset -= mean
    dataset /= std
    dataset = np.vstack((dataset, dataset_labels))

    if other is not None:
        other = (other - mean) / std
        other = np.vstack((other, other_labels))
        return (dataset, other)

    return dataset


def gaussianize(dataset : np.ndarray , other : np.ndarray = None, feat_label : bool = False) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
    '''
    Write me
    '''
    shape = dataset.shape

    if (len(shape) < 2):
        r, c = 1, len(dataset)
        dataset = dataset.reshape((r, c))
    else:
        r, c = dataset.shape
    gdataset = np.zeros((r, c))


    for feat_idx in range(r):
        ranks = np.zeros(c)
        feats = dataset[feat_idx, :]
        for idx in range(c):
            value = feats[idx]
            ranks[idx] = ((feats < value).sum() + 1) / (c+2)
            ranks[idx] = ppf(ranks[idx])
        gdataset[feat_idx] = ranks

    if other is not None:
       otherr, otherc = other.shape
       out = np.empty((otherr, otherc))
       for feat_idx in range(r):
           dfeats = dataset[feat_idx]
           ofeats = other[feat_idx]
           ranks = np.zeros(otherc)
           for idx in range(otherc):
               value = ofeats[idx]
               ranks[idx] = ((dfeats < value).sum() + 1) / (c+2)
               ranks[idx] = ppf(ranks[idx])
           out[feat_idx] = ranks
       return gdataset, out

    return gdataset

def support_vectors(dataset : np.ndarray, alphas : np.array, zero : float = .0) -> Tuple[np.ndarray, np.array]:
    '''
    Extracts only support vectors from a trained dual SVM model
    '''
    selector = alphas > zero
    alphas = alphas[selector]
    dataset = dataset[:, selector]
    return dataset, alphas

def whiten(dataset : np.ndarray) -> np.ndarray:
    '''
    Returns the eigenvalues `w` and eigenvectors `v` used to whiten a dataset
    '''

    dataset = normalize(dataset)
    feats, labels = dataset[:-1, :], dataset[-1, :]
    cov = fc_cov(feats)
    w, v = PCA(cov, feat_label=False)
    w[w < 1e-10] = 1e-10
    w = np.diag(w)
    w = sp.linalg.fractional_matrix_power(w, -0.5)
    pcov = w @ cov @ w
    w, v = PCA(pcov, feat_label=False)
    return w, v


def BEP(scores, labels, N=100):
    pis = np.linspace(0.01, 0.99, N)
    mindcf_points, actdcf_points, xaxis = [], [], []
    _, opt = minDCF(scores, labels)

    for pi in pis:
        t = np.log( pi/(1-pi) )
        mindcf, _ = DCF(scores > opt, labels, prior_t=pi)
        dcf, _ = DCF(scores > -t, labels, prior_t=pi)

        mindcf_points.append(mindcf)
        actdcf_points.append(dcf)
        xaxis.append(t)

    return (actdcf_points, xaxis), (mindcf_points, xaxis)

def calibrate_scores(scores, labels, p):
    from classifiers import logreg
    data = np.vstack([scores, labels])
    alpha, beta = logreg(data)
    return alpha*scores + beta - np.log(p/(1-p))