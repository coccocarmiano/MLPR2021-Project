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
    plt.scatter(good[0], good[1], color=utils.green)
    plt.scatter(bad[0], bad[1], color=utils.red)   


    plt.figure()
    data = utils.normalize(data)
    feats, labels = data[:-1, :], data[-1, :]
    mean = utils.fc_mean(feats)
    w, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    good, bad = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(good[0], good[1], color=utils.green)
    plt.scatter(bad[0], bad[1], color=utils.red)   


    plt.figure()
    data = utils.load_train_data()
    w, v = utils.whiten(data)
    feats, labels = data[:-1, :], data[-1, :]
    feats = feats - utils.fc_mean(feats)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    good, bad = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(good[0], good[1], color=utils.green)
    plt.scatter(bad[0], bad[1], color=utils.red)    
    plt.show()