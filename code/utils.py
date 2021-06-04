import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_data_file = '../data/Train.txt'
nfeats = 11

def load_train_data():
    '''
    Returns data from `Train.txt` organized as column samples.
    Labels > 6 are converted to 1.
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
    Params:

    `dataset` Dataset to analyze.
    Assumes last feature to be label.

    `stats` Default `False`. If true prints some stats about
    the dataset eigenvalues.

    Returns:

    `w` Descending order sorted egienvalues

    `v` Corresponding normalized eigenvectors
    '''

    feats = dataset[:-1, :]

    f_means = feats.mean(axis=1).reshape((11, 1))
    cent = feats -f_means
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
    
