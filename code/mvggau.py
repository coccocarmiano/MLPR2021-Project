import numpy as np
from numpy import diag
from classifiers import gaussian_classifier
from matplotlib import pyplot as plt
import utils

def fullcov(folds, priors, npca, eigv):
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            gcov = utils.fc_cov(trs[:, trl > 0])
            bcov = utils.fc_cov(trs[:, trl < 1])

            pgmean = vt.T @ gmean
            pbmean = vt.T @ bmean

            pgcov = vt.T @ gcov @ vt
            pbcov = vt.T @ bcov @ vt

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pbcov, pgcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        for prior in priors:
            mindcf, t = utils.minDCF(scores, labels, prior_t=prior)
            print(f"{mindcf} | Effective Prior: {prior} | Optimal Threshold: {t} | PCA {n}")


def naive(folds, priors, npca, eigv):
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            gcov = utils.fc_cov(trs[:, trl > 0])
            bcov = utils.fc_cov(trs[:, trl < 1])

            pgmean = vt.T @ gmean # Diff
            pbmean = vt.T @ bmean # Diff

            pgcov = np.eye(n) * (vt.T @ gcov @ vt)
            pbcov = np.eye(n) * (vt.T @ bcov @ vt)

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pbcov, pgcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        
        for prior in priors:
            mindcf, t = utils.minDCF(scores, labels, prior_t=prior)
            print(f"{mindcf} | Effective Prior: {prior} | Optimal Threshold: {t} | PCA {n}")


def tied(folds, priors, npca, eigv):
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            cov = utils.fc_cov(trs) # Diff

            pgmean = vt.T @ gmean
            pbmean = vt.T @ bmean

            pcov = vt.T @ cov @ vt

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pcov, pcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        for prior in priors:
            mindcf, t = utils.minDCF(scores, labels, prior_t=prior)
            print(f"{mindcf} | Effective Prior: {prior} | Optimal Threshold: {t} | PCA {n}")


if __name__ == '__main__':
    toprint = []  # ignore this var
    to_plot = []
    trdataset = utils.load_train_data()
    trd, trl = trdataset[:-1, :], trdataset[-1, :]
    trd = utils.gaussianize(trdataset)
    trdataset = np.vstack((trd, trl))
    nfolds = 20

    _, v = utils.PCA(trdataset)
    _, folds = utils.kfold(trdataset, n=nfolds)

    priors = [.1, .5, .9]
    npca = np.arange(11)+1

    fullcov(folds, priors, npca, v)
    #naive(folds, priors, npca, v)
    #tied(folds, priors, npca, v)

    feats, labels = trdataset[:-1, :], trdataset[-1, :]
    feats = utils.normalize(feats)
    trdataset = np.vstack((feats, labels))

    _, v = utils.PCA(trdataset)
    _, folds = utils.kfold(trdataset, n=nfolds)
    
    #fullcov(folds, priors, npca, v)
    #naive(folds, priors, npca, v)
    #tied(folds, priors, npca, v)