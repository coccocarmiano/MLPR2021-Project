import numpy as np
import matplotlib.patches as ptc


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


def PCA(dataset, stats=False):
    '''
    Execute eigenvalue decompsition on `dataset`.

    Parameters:
    
    `dataset`: numpy matrix of column samples. Assumes last feat to label.

    `stats` (opt): Print some stats like information retention and (soon) correlation matrix.

    Returns:
    
    The (already sorted) array `w` of eigenvalues and `v` of eigenvectors.
    '''

    feats = dataset[:-1, :]

    f_means = feats.mean(axis=1).reshape((11, 1))
    cent = feats - f_means
    mult = cent.dot(cent.T) / dataset.shape[1]
    w, v = np.linalg.eigh(mult)
    w, v = w[::-1], v[:, ::-1]
    if stats:
        wsum = sum(w)
        for i in range(nfeats):
            print(f"{i+1} Features: {sum(w[:i+1])/wsum*100:.4f}%")
        print("Eigenvalues: ", w)

        # Todo: Covariance to Correlation
    return w, v


def get_patches():
    '''
    Returns: patches list to be passed to either `legend` or `figlegend` (to the `handles` attribute).
    '''
    bpatch = ptc.Patch(color=red, label="Bad Wine")
    gpatch = ptc.Patch(color=green, label="Good Wine")
    return [gpatch, bpatch]


def kfold(dataset : np.ndarray, n : int=5) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
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
    
    splits = np.array(splits)

    folds = []
    sel = np.arange(n)
    for i in range(n):
        train, test = splits[sel != i], splits[sel == i]
        train = np.concatenate(train[:], axis=1)
        test = np.concatenate(test[:], axis=1)
        folds.append((train, test))
    splits = [split for split in splits]
    return splits, folds