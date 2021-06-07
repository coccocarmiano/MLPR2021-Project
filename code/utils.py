import numpy as np
import matplotlib.patches as ptc
from typing import List, Tuple

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
    train_file = open(test_data_file, 'r')
    lines = [line.strip() for line in train_file]
    train_file.close()

    splits = []
    for line in lines:
        split = line.split(',')
        arr = np.array([float(i) for i in split])
        splits.append(arr)
    matrix = np.array(splits).T
    return matrix


def PCA(dataset: np.ndarray, feat_label: bool = True, stats: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Execute eigenvalue decompsition on `dataset`.

    Parameters:

    `dataset`: numpy matrix of column samples. Assumes last feat to label.

    `feat_label`: if false assumes no label feature

    `stats` (opt): Print some stats like information retention and (soon) correlation matrix.

    Returns:

    The (already sorted) array `w` of eigenvalues and `v` of eigenvectors.
    '''

    if feat_label:
        feats = dataset[:-1, :]
    else:
        feats = dataset

    _, c = dataset.shape
    f_means = fc_mean(feats)
    cent = feats - f_means
    mult = (cent @ cent.T) / c
    w, v = np.linalg.eigh(mult)
    w, v = w[::-1], v[:, ::-1]
    if stats:
        wsum = sum(w)
        for i in range(nfeats):
            print(f"{i+1} Features: {sum(w[:i+1])/wsum*100:.4f}%")
        print("Eigenvalues: ", w)

        # Todo: Covariance to Correlation
    return w, v


def get_patches() -> List[ptc.Patch]:
    '''
    Returns: patches list to be passed to either `legend` or `figlegend` (to the `handles` attribute).
    '''
    bpatch = ptc.Patch(color=red, label="Bad Wine")
    gpatch = ptc.Patch(color=green, label="Good Wine")
    return [gpatch, bpatch]


def kfold(dataset: np.ndarray, n: int = 5) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    '''
    Splits `dataset` in `n` folds. Returns a tuple.

    First list contains `n` evenly divided subsets of `dataset`.

    Second list contains `n` tuples. Each tuple has `n-1` folds
    of samples from `dataset` and the the remaining `1`.
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
        train = np.concatenate(train[:], axis=1)
        test = np.concatenate(test[:], axis=1)
        folds.append((train, test))
    splits = [split for split in splits]
    return splits, folds


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
    dmean = fc_mean(dataset)
    centered = dataset - dmean
    cov = centered.dot(centered.T) / c
    return cov


def DCF(predictions: np.ndarray, labels: np.ndarray, prior_t: float = 0.5, costs: Tuple[float, float] = (1., 1.)) -> float:
    '''
    Returns the normalized and unnormalized DCF values

    `predictions` are the assigned labels after classificaton

    `labels` are the real labels

    `prior_t` is the prior probability of class T

    `costs` is a tuple containing FIRST the cost for misclassifying as F an elem of class T, then the other
    '''
    FPR = ((predictions == 1) == (labels == 0)).sum() / len(predictions)
    FNR = ((predictions == 0) == (labels == 1)).sum() / len(predictions)
    unnorm_dcf = FNR*costs[0]*prior_t + FPR * costs[1] * (1-prior_t)
    factr = min(prior_t * costs[0], (1-prior_t) * costs[1])
    norm_dcf = unnorm_dcf / factr

    return (norm_dcf, unnorm_dcf)


def normalize(dataset: np.ndarray, other: np.ndarray = None, has_labels=False) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
    '''
    Z-Normalize a (two) dataset(s).

    If `other` is provided, normalizes it with the data from `dataset` and returns a tuple with normalized
    `dataset, other`, otherwise just normalized dataset.

    If `has_labels` is `True`, discars label feature (assumed last one).
    '''
    if has_labels:
        dataset = dataset[:-1, :]
        if other is not None:
            other = other[:-1, :]

    r, _ = dataset.shape
    mean = fc_mean(dataset)
    std = dataset.std(axis=1).reshape((r, 1))
    dataset -= mean
    dataset /= std

    if other is not None:
        other = (other - mean) / std
        return (dataset, other)

    return dataset
