import numpy as np
import utils
import scipy as sp
from matplotlib import pyplot as plt

fout = '../img/scatter.jpg'
nfeats = 11

if __name__ == '__main__':
    data = utils.load_train_data()
    feats, labels = data[:-1, :], data[-1, :]
    mean = utils.fc_mean(feats)
    w, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    good, bad = feats[:, labels > 0], feats[:, labels < 1]
    plt.title("Raw Features, PCA 2")
    plt.scatter(good[0], good[1], marker='x', color=utils.green)
    plt.scatter(bad[0], bad[1], marker='x', color=utils.red)
    plt.savefig('../img/2DRAW')


    plt.figure()
    plt.title("Normalized Features, PCA 2")
    data = utils.normalize(data)
    feats, labels = data[:-1, :], data[-1, :]
    mean = utils.fc_mean(feats)
    w, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    good, bad = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(good[0], good[1], marker='x', color=utils.green)
    plt.scatter(bad[0], bad[1], marker='x', color=utils.red)
    plt.savefig('../img/2DNorm')   


    plt.figure()
    plt.title("Whitened Features, PCA 2")
    data = utils.load_train_data()
    data = utils.normalize(data)
    w, v = utils.PCA(data)
    feats, labels = data[:-1, :], data[-1, :]
    vt, wt = v[:, :2], w[:2]
    S = np.diag(wt ** -0.5)
    pfeats = vt.T @ feats
    pfeats = S @ pfeats
    good, bad = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(good[0], good[1], marker='x', color=utils.green)
    plt.scatter(bad[0], bad[1], marker='x', color=utils.red)    

    plt.figure()
    plt.title("Whitened Features")
    data = utils.load_train_data()
    data = utils.normalize(data)
    w, v = utils.whiten(data)
    feats, labels = data[:-1, :], data[-1, :]
    vt, wt = v[:, :2], w[:2]
    pfeats = vt.T @ feats
    good, bad = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(good[0], good[1], marker='x', color=utils.green)
    plt.scatter(bad[0], bad[1], marker='x', color=utils.red)    
    plt.savefig('../img/2DWhitened')