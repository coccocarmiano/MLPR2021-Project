import numpy as np
from numpy import pi, log, exp, diag
from classifiers import gaussian_classifier
import utils

def latex(toprint):
    outfiletex = '../data/mvgn_naiveacctable.tex'
    f = open(outfiletex, "w")
    print(r"\begin{tabular}{|c|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"$\pi_T$ & PCA & Error Rate & $DCF_{norm}$\\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[3])
    toprint = toprint[:10]
    for tup in toprint:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% & {tup[3]:.3f}\\\\", file=f)
        print(r"\hline", file=f)

    print(r"\end{tabular}", file=f)
    print(r"\caption{Naive Bayes MVG (Normalized Samples)}\label{tab:mvgn_naiveacctable}", file=f)

    f.close()

if __name__ == '__main__':
    toprint = [] #ignore this var
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()

    trsamp, trlab = trdataset[:-1, :], trdataset[-1, :]
    tesamp, telab = tedataset[:-1, :], tedataset[-1, :]
    trsamp, tesamp = utils.normalize(trsamp, other=tesamp)

    gmean, bmean = utils.fc_mean(trsamp[:, trlab > 0]), utils.fc_mean(trsamp[:, trlab < 1])
    gcov, bcov = utils.fc_cov(trsamp[:, trlab > 0]), utils.fc_cov(trsamp[:, trlab < 1])
    gcov, bcov = diag(diag(gcov)), diag(diag(bcov))

    r, c = trsamp.shape

    it = np.arange(r)
    it = it[-1:0:-1]

    w, v = utils.PCA(trdataset)
    priors = (np.arange(9)+1)/10
    
    for prior in priors:
        for n in it:
            vt = v[:, :n]

            pdata = vt.T @ tesamp
            pgmean, pbmean = vt.T @ gmean, vt.T @ bmean
            pgcov, pbcov = vt.T @ gcov @ vt, vt.T @ bcov @ vt

            scores, predictions = gaussian_classifier(pdata, [pbmean, pgmean], [pbcov, pgcov], prior_t=prior)

            nt = len(predictions)
            nc = (predictions == telab).sum()
            acc = nc/nt

            dcf, _ = utils.DCF(predictions, telab, prior_t=prior)
            toprint.append((prior, n, acc, dcf))


    latex(toprint)
