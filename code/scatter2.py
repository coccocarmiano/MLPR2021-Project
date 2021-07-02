from matplotlib import pyplot as plt
import numpy as np
import utils

if __name__ == '__main__':
    train, test = utils.load_train_data(), utils.load_test_data()
    train, test = utils.normalize(train, other=test)
    w, v = utils.whiten(train)
    vt = v[:, :2]
    trfeats, tefeats = train[:-1, :], test[:-1, :]
    trlab, telab = train[-1, :], test[-1, :]

    plt.figure()
    ptrfeats = vt.T @ trfeats
    ptefeats = vt.T @ tefeats

    plt.scatter(ptrfeats[0, trlab < 1], ptrfeats[1, trlab < 1], alpha=.3, color='red')
    plt.scatter(ptrfeats[0, trlab > 0], ptrfeats[1, trlab > 0], alpha=.3, color='green')

    plt.scatter(ptefeats[0, telab < 1], ptefeats[1, telab < 1], marker='x',color='red')
    plt.scatter(ptefeats[0, telab > 0], ptefeats[1, telab > 0], marker='x',color='green')
    plt.show()

    """ test = utils.load_test_data()
    w, v = utils.whiten(test)
    feats, labels = test[:-1, :], test[-1, :]
    vt = v[:, :2]
    pfeats = vt.T @ feats
    gf, bf = pfeats[:, labels > 0], pfeats[:, labels < 1]
    plt.scatter(gf[0], gf[1], marker='x', color='green')
    plt.scatter(bf[0], bf[1], marker='x', color='red')
    plt.show() """

    """ train = utils.load_train_data()[5:7, :]
    plt.scatter(train[0], train[1], marker='x', color="red")
    w, v = utils.PCA(train, feat_label=False)
    print(w, v)
    train = v.T @ train
    plt.scatter(train[0], train[1], marker='x', color="blue")
    plt.show() """

    """ train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    print(v.shape)
    vt = v[:, :2]
    w = w[:2]
    train = utils.normalize(train)
    feats, labels = train[:-1, :], train[-1, :]
    pfeats = vt.T @ feats
    cov1 = utils.fc_cov(pfeats)
    plt.scatter(pfeats[0, labels < 1], pfeats[1, labels < 1], marker='x', color='orange')
    plt.scatter(pfeats[0, labels > 0], pfeats[1, labels > 0], marker='x', color='blue')
    Z = np.diag(w ** -0.5) @ pfeats
    plt.scatter(Z[0, labels < 1], Z[1, labels < 1], marker='x', color='red')    
    plt.scatter(Z[0, labels > 0], Z[1, labels > 0], marker='x', color='green')    
    cov2 = utils.fc_cov(Z)
    plt.show()
 """