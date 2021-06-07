import numpy as np
from numpy import pi, log, exp
from classifiers import gaussian_classifier
import utils

def latex(toprint):
    outfiletex = '../data/mvg_acctable.tex'
    f = open(outfiletex, "w")
    print(r"\caption{MVG}\label{tab:mvg_acctable}", file=f)
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"$\pi_T$ & PCA & Error Rate & $DCF_{min}$\\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[3])
    toprint = toprint[:10]
    for tup in toprint:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% & {tup[3]:.3f}\\\\", file=f)
        print(r"\hline", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)

    f.close()

if __name__ == '__main__':
    toprint = [] #ignore this var
    trdataset = utils.load_train_data()

    _, folds = utils.kfold(trdataset)

    priors = [.33, .5, .67]
    pca_it = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for fold in folds:
        trs, trl = fold[0][:-1, :], fold[0][-1, :]
        tes, tel = fold[1][:-1, :], fold[1][-1, :]

        gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
        gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])

        w, v = utils.PCA(trs, feat_label=False)

        for prior in priors:
            for n in pca_it:
                vt = v[:, :n]
                proj = vt.T @ tes
                pgmean, pbmean = vt.T @ gmean, vt.T @ bmean
                pgcov, pbcov = vt.T @ gcov @ vt, vt.T @ bcov @ vt
                scores, predictions = gaussian_classifier(proj, [pbmean, pgmean], [pbcov, pgcov], prior_t=prior)

                nt = len(predictions)
                nc = (predictions == tel).sum()
                acc = nc/nt
                dcf, _ = utils.DCF(predictions, tel, prior_t=prior)
                toprint.append((prior, n, acc, dcf))

    latex(toprint)
