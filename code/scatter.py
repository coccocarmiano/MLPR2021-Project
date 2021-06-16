import numpy as np
import utils
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

    feats = utils.normalize(data)[:-1, :]
    _, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    x, y = pfeats[0], pfeats[1]
    plt.subplot(3, 1, 2)
    plt.scatter(x[labels > 0], y[labels > 0], alpha = .75, color=utils.green)
    plt.scatter(x[labels < 1], y[labels < 1], alpha = .75, color=utils.red)


    feats = utils.normalize(data)
    feats = utils.whiten(feats)[:-1, :]
    _, v = utils.PCA(feats, feat_label=False)
    vt = v[:, :2]
    pfeats = vt.T @ feats
    x, y = pfeats[0], pfeats[1]
    plt.subplot(3, 1, 3)
    plt.scatter(x[labels > 0], y[labels > 0], alpha = .75, color=utils.green)
    plt.scatter(x[labels < 1], y[labels < 1], alpha = .75, color=utils.red)
    #plt.savefig(fout, format='jpg')
    plt.show()
