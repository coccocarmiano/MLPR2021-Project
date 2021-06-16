import numpy as np
from classifiers import gaussian_classifier
import utils

if __name__ == '__main__':
    train = utils.load_train_data()
    train_s, train_l = train[:-1, :], train[-1, :]

    test = utils.load_test_data()
    test_s, test_l = test[:-1, :], test[-1, :]

    gtrain_s, gtest_s = utils.gaussianize(train_s, other=test_s)

    _, v = utils.PCA(gtrain_s, feat_label=False)
    vt = v[:, :2]

    ptrain, ptest = vt.T @ gtrain_s, vt.T @ gtest_s
    gmean, bmean = utils.fc_mean(ptrain[:, train_l > 0]), utils.fc_mean(ptrain[:, train_l < 1])
    gcov, bcov = utils.fc_cov(ptrain[:, train_l > 0]), utils.fc_cov(ptrain[:, train_l < 1])
    print(gcov.shape, bcov.shape)
    scores, _ = gaussian_classifier(ptest, [bmean, gmean], [bcov, gcov])
    pred = scores > -0.0333564
    nc = (pred == test_l).sum()
    nt = (len(pred))
    acc = nc/nt*100
    print(f"Accuracy: {acc:.3f}%")