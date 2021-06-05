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
