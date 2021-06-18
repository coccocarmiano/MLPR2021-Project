import numpy as np
import utils
import scipy as sp
from matplotlib import pyplot as plt

fout = '../img/scatter.jpg'
nfeats = 11

if __name__ == '__main__':
    data = utils.load_train_data()
    feats, labels = data[:-1, :], data[-1, :]
    _, v = utils.PCA(data)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    x, y = pfeats[0], pfeats[1]
    plt.subplot(3, 1, 1)
    plt.scatter(x[labels > 0], y[labels > 0], alpha = .75, color=utils.green)
    plt.scatter(x[labels < 1], y[labels < 1], alpha = .75, color=utils.red)
    plt.title("PCA 2, Raw Feats")

    feats = utils.normalize(data)[:-1, :]
    cov = utils.fc_cov(feats)
    _, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    x, y = pfeats[0], pfeats[1]
    plt.subplot(3, 1, 2)
    plt.scatter(x[labels > 0], y[labels > 0], alpha = .75, color=utils.green)
    plt.scatter(x[labels < 1], y[labels < 1], alpha = .75, color=utils.red)
    plt.title("PCA 2, Normalized Feats")


    feats = utils.normalize(data)
    feats = data[:-1, :]
    feats -= utils.fc_mean(feats)
    w, v = utils.PCA(feats, feat_label=False)
    sigma = np.diag(w)
    invsig = sp.linalg.fractional_matrix_power(sigma, -0.5)
    P = v @ invsig @ v.T
    feats = P @ feats
    pfeats = feats

    w, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    x, y = pfeats[0], pfeats[1]
    plt.subplot(3, 1, 3)
    plt.scatter(x[labels > 0], y[labels > 0], alpha = .75, color=utils.green)
    plt.scatter(x[labels < 1], y[labels < 1], alpha = .75, color=utils.red)
    plt.title("PCA 2, Normalized + Whitend Feats")
    #plt.savefig(fout, format='jpg')
    plt.show()
